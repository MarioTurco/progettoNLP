import networkx as nx
from datetime import datetime, timedelta
from number_parser import parse_number
import tensorflow as tf
from collections import OrderedDict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

global_node_id = 2


class JointIntentAndSlotFillingModel_v2(tf.keras.Model):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name="model", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.hidden_layer = Dense(128, activation="relu")
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        # two outputs from BERT
        trained_bert = self.bert(inputs, **kwargs)
        pooled_output = trained_bert.pooler_output
        sequence_output = trained_bert.last_hidden_state

        # sequence_output will be used for slot_filling / classification
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get("training", False))
        sequence_output = self.hidden_layer(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        # pooled_output for intent classification
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        pooled_output = self.hidden_layer(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits

class JointIntentAndSlotFillingModel(tf.keras.Model):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name="model", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        # two outputs from BERT
        trained_bert = self.bert(inputs, **kwargs)
        pooled_output = trained_bert.pooler_output
        sequence_output = trained_bert.last_hidden_state
        
        # sequence_output will be used for slot_filling / classification
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)

        # pooled_output for intent classification
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits

class DatasetPreprocessor:
    def encode_slots(self, all_slots, all_texts, 
                     tokenizer, slot_map, max_len):
        encoded_slots = np.zeros(shape=(len(all_texts), max_len), dtype=np.int32)
        
        for idx, text in enumerate(all_texts):
            enc = [] # for this idx, to be added at the end to encoded_slots
            
            # slot names for this idx
            slot_names = all_slots[idx]
            
            # raw word tokens
            # not using bert for this block because bert uses
            # a wordpiece tokenizer which will make 
            # the slot label to word mapping
            # difficult
            raw_tokens = text.split()

            # words or slot_values associated with a certain
            # slot_name are contained in the values of the
            # dict slots_names
            # now this becomes a two way lookup
            # first we check if a word belongs to any
            # slot label or not and then we add the value from
            # slot map to encoded for that word
            for rt in raw_tokens:
                # use bert tokenizer
                # to get wordpiece tokens
                bert_tokens = tokenizer.tokenize(rt)
                
                # find the slot name for a token
                rt_slot_name = self.get_slot_from_word(rt, slot_names)
                if rt_slot_name is not None:
                    # fill with the slot_map value for all ber tokens for rt
                    enc.append(slot_map[rt_slot_name])
                    enc.extend([slot_map[rt_slot_name]] * (len(bert_tokens) - 1))

                else:
                    # rt is not associated with any slot name
                    enc.append(0)

            
            # now add to encoded_slots
            # ignore the first and the last elements
            # in encoded text as they're special chars
            encoded_slots[idx, 1:len(enc)+1] = enc
        
        return encoded_slots

    def get_slot_map(self, slot_label):
        slot_map = dict() # slot -> index
        for idx, us in enumerate(slot_label):
            slot_map[us] = idx
        return slot_map

    # gets slot name from its values
    def get_slot_from_word(self, word, slot_dict):
        for slot_label,value in slot_dict.items():
            if word in value.split():
                return slot_label
        return None

    def preprocess_slot_labels(self, unique_slot_labels):
        slot_label = ['O']
        with open(unique_slot_labels, 'r') as f:
            for line in f.readlines()[3:]:
                slot_label.append(line.rstrip('\n')[2:])
        slot_label = list(OrderedDict.fromkeys(slot_label))
        slot_label.insert(0, '<PAD>')
        return slot_label

    def read_intent_labels(self, intent_labels_path):
        intent_labels = []
        with open(intent_labels_path, 'r') as f:
            for line in f.readlines()[1:]:
                intent_labels.append(line.rstrip('\n'))
        return intent_labels

    # Costruisce per ogni frase un dizionario con chiave nome dello slot e valore valore dello slot nella frase
    def get_slot_indexes(self, text, slots):
        slot_list = []
        for index, bio_slot in enumerate(slots):
            split_text = text[index].split()
            slots = {}
            split_bio_slot = bio_slot.split()
            for ind, slot in enumerate(split_bio_slot):
                if slot[:2] == "B-":
                    start = end = ind
                    while (end+1 < len(split_bio_slot)) and split_bio_slot[end+1][:2] == "I-":
                        end += 1
                    if end < len(split_bio_slot):
                        end += 1
                    slots[slot[2:]] = " ".join(split_text[start:end])
            slot_list.append(slots)
        return slot_list

    def read_file(self, text_path):
        lines = []
        with open(text_path, 'r') as f:
            for line in f.readlines():
                lines.append(line.rstrip('\n'))
        return lines
    
    def encode_texts(self, tokenizer, texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

    def get_intent_map(self, intent_labels):
        intent_map = dict() # index -> intent
        for idx, ui in enumerate(intent_labels):
            intent_map[ui] = idx
        return intent_map

    # map to train_data values
    def encode_intents(self, intents, intent_map):
        encoded = []
        for i in intents:
            encoded.append(intent_map[i])
        # convert to tf tensor
        return tf.convert_to_tensor(encoded, dtype="int32")

class MetricsCalculator:
    def __init__(self):
        MetricsCalculator.accuracy_metrics = staticmethod(MetricsCalculator.accuracy_metrics)
        MetricsCalculator.precision_recall_fscore_metrics = staticmethod(MetricsCalculator.precision_recall_fscore_metrics)

    def accuracy_metrics(intents_true, slots_true, x_test, model):
        m = SparseCategoricalAccuracy("accuracy")
        m.reset_state()
        m.update_state(intents_true, model(x_test)[1])
        print("Acc intents: " + str(m.result().numpy()))
        m.reset_state()
        m.update_state(slots_true, model(x_test)[0])
        print("Acc slots: " + str(m.result().numpy()))

    #returns accuracy, precision, recall, fscore
    def precision_recall_fscore_metrics(intents_true, slots_true, x_test, model):
        intent_id = model(x_test)[1].numpy().argmax(axis=-1)
        print("Precision Recall F1 intents: " + str(precision_recall_fscore_support(intents_true, intent_id, average='macro')[:3]))
        m = MultiLabelBinarizer().fit(slots_true)
        slots_id = model(x_test)[0].numpy().argmax(axis=-1)
        print("Precision Recall F1 slots: " + str(precision_recall_fscore_support(m.transform(slots_true), m.transform(slots_id), average='macro')[:3]))
    

class Dialog:
    def __init__(self, user_name, user_id, classification_model, intent_labels, slot_labels, tokenizer):
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels
        self.graph = nx.DiGraph()
        self.user_name = user_name
        self.user_id = user_id
        self.dialog_id = 1
        self.start_dialog = datetime.now()
        self.graph, self.dt_string = self.init_dialog(self.user_id, self.user_name, self.dialog_id, self.start_dialog)
        self.joint_model = classification_model
        self.book_intent_processor = BooksIntentProcessor()
        self.weather_intent_processor = WeatherIntentProcessor()
        self.restaurant_intent_processor = RestaurantIntentProcessor()
        self.intent_processors = {'GetWeather' : self.weather_intent_processor.create_weather_references,
                                  'BookRestaurant' : self.restaurant_intent_processor.create_restaurant_references,
                                  'RateBook' : self.book_intent_processor.create_book_rating_references}
        self.last_node = 1
        self.tokenizer = tokenizer

        Dialog.check_true_values = staticmethod(Dialog.check_true_values)

    def nlu(self, text, tokenizer, model):
        inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
        outputs = model(inputs)
        slot_logits, intent_logits = outputs
        slot_ids = slot_logits.numpy().argmax(axis=-1)[0, :]
        intent_id = intent_logits.numpy().argmax(axis=-1)[0]

        info = {"intent": self.intent_labels[intent_id], "slots": {}}
        out_dict = {}
        # get all slot names and add to out_dict as keys
        predicted_slots = set([self.slot_labels[s] for s in slot_ids if s != 0])
        for ps in predicted_slots:
            out_dict[ps] = []

        # check if the text starts with a small letter
        if text[0].islower():
            tokens = tokenizer.tokenize(text, add_special_tokens=True)
        else:
            tokens = tokenizer.tokenize(text)
        for token, slot_id in zip(tokens, slot_ids):
            # add all to out_dict
            slot_name = self.slot_labels[slot_id]
            if slot_name == "<PAD>":
                continue

            # collect tokens
            collected_tokens = [token]
            idx = tokens.index(token)

            # see if it starts with ##
            # then it belongs to the previous token
            if token.startswith("##"):
                # check if the token already exists or not
                if tokens[idx - 1] not in out_dict[slot_name]:
                    collected_tokens.insert(0, tokens[idx - 1])
            # add collected tokens to slots
            out_dict[slot_name].extend(collected_tokens)

        # process out_dict
        for slot_name in out_dict:
            tokens = out_dict[slot_name]
            slot_value = tokenizer.convert_tokens_to_string(tokens)

            info["slots"][slot_name] = slot_value.strip()

        return info

    def check_true_values(slots, slot_names_list):
        true_slots = {}
        for name in slot_names_list:
            if name in slots:
                true_slots[name] = slots[name]
            else:
                true_slots[name] = None
        return true_slots


    def init_dialog(self, user_id, user_name, dialog_id, now):
        graph = nx.DiGraph()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        graph.add_node(user_id, name=user_name, labels=":User")
        graph.add_node(dialog_id, labels=":Dialog")
        graph.add_edge(user_id, dialog_id, label="PARTECIPATES_TO")
        #nodo data
        graph.add_node(dt_string, labels=":Date", date=now.strftime("%d/%m/%Y"), time=now.strftime("%H:%M:%S"))
        graph.add_edge(dialog_id, dt_string, label="TAKES_PLACE_ON")
        return graph, dt_string
    
    def create_intent_and_references(self, utterance, tokenizer):
        global global_node_id
        info = self.nlu(utterance, tokenizer, self.joint_model)
        text_encoding = global_node_id
        self.graph.add_node(text_encoding, labels=":Utterance", text=utterance)
        global_node_id += 1
        if self.graph.nodes[self.last_node]['labels'] == ':Dialog':
            self.graph.add_edge(self.last_node, text_encoding, label='STARTS_WITH')
        else:
            self.graph.add_edge(self.last_node, text_encoding, label='FOLLOWED_BY')
        self.graph.add_edge(text_encoding, self.last_node, label='FOLLOWS')

        self.create_intent(info['intent'], text_encoding)
        #print(info['intent'])
        if info['intent'] not in self.intent_processors:
            print("Sorry, I can't do that")
        else:
            self.intent_processors[info['intent']](info['slots'], self.graph, text_encoding, self.user_id, start_dialog=self.start_dialog)
        self.last_node = text_encoding

    def create_intent(self, intent, text_encoding):
        self.graph.add_node(intent, labels=":Intent", intent=intent)
        self.graph.add_edge(text_encoding, intent, label="EXPRESSES_INTENT")

    def interact(self, utterance):
        self.create_intent_and_references(utterance, self.tokenizer)
    
    def save_graph(self, save_path='example.graphml'):
        nx.write_graphml(self.graph, save_path, named_key_ids=True)
    '''
    def interact(self, utterance, tokenizer):
        self.create_intent_and_references(utterance, tokenizer)
    '''
class BooksIntentProcessor:
    def __init__(self):
        self.rating_slot_names = ['object_name', 'rating_value', 'object_select', 'object_type', 'best_rating', 'rating_unit', 'object_part_of_series_type']
        self.book_slot_names = ['object_type', 'object_part_of_series_type']
    
    # Se rating e best_value sono validi, allora si controlla se rating <= best_value. Se il best_value non é valido, allora si aggiunge solo il rating
    def add_rating_to_book(self, slots, rating_slots, graph, user, utterance_node_id):
        #print(rating_slots)
        book_dict = {}
        for prop in self.book_slot_names:
            if prop in slots:
                book_dict[prop] = slots[prop]

        graph.add_node(rating_slots['object_name'], labels=":Book", **book_dict)
        if not rating_slots['rating_value']:
            val = -1
        else:
            val = parse_number(rating_slots['rating_value'].split(' ')[0])
        if rating_slots['rating_value'] and val >= 0:
            #print(val, 'out of', rating_slots['best_rating'])
            if rating_slots['best_rating'] and val >= 0:
                best_val = parse_number(rating_slots['best_rating'].split(' ')[0])
                if val <= best_val:
                    graph.add_edge(user, rating_slots['object_name'], label='RATES', rating=val, best_value=best_val)
                    print("Rating added for ", rating_slots['object_name'])
                else:
                    print("You cannot rate a book more than the best rating")
            else:
                graph.add_edge(user, rating_slots['object_name'], label='RATES', rating=val)
                print("Rating added for ", rating_slots['object_name'])
        else:
            print('Invalid rating value ')
        graph.add_edge(utterance_node_id, rating_slots['object_name'], label='REFERS_TO')
  
    def create_book_rating_references(self, slots, graph, utterance_node_id, user_id, **kwargs):
        rating_slots = Dialog.check_true_values(slots, self.rating_slot_names)     # Costruisco un dizionario con tutti i possibili slot, in modo tale da controllare facilmente se gli slot sono riempiti
        if rating_slots['object_name']:
            self.add_rating_to_book(slots, rating_slots, graph, user_id, utterance_node_id)
        # Se il nome non é specificato o si indica 'current' allora si cerca l'ultimo libro (di un determinato tipo se specificato)
        elif (not rating_slots['object_name'] or slots['object_select'] == 'current'):
            self.rate_last_book(graph, utterance_node_id, slots, rating_slots, user_id)

    def find_last_book(self, node, graph, object_type=None):
        if graph.nodes[node]['labels'] == ':Dialog':
            return -1
        next_node = None
        for neigh in graph.neighbors(node):
            if ((object_type and \
                'object_type' in graph.nodes[neigh] and \
                graph.nodes[neigh]['object_type'] == object_type) \
            or \
            not object_type) and \
            graph.nodes[neigh]['labels'] == ':Book' :
                return neigh
            # In un dialogo, per ogni utterance puó esserci un solo arco follows
            if graph[node][neigh]['label'] == 'FOLLOWS' and (graph.nodes[neigh]['labels'] == ':Utterance' or graph.nodes[neigh]['labels'] == ':Dialog'):
                next_node = neigh
        return self.find_last_book(next_node, graph, object_type)    

    def rate_last_book(self, graph, utterance_node_id, slots, rating_slots, user): #Se object_type=None prende semplicemente l'ultimo libro presente nel dialogo
        last_book = self.find_last_book(utterance_node_id, graph, rating_slots['object_type'])
        if last_book == -1:
            print('No book found, please rephrase specifying a book')
        else:
            rating_slots['object_name'] = last_book
            #print("Book found: " + last_book)
            self.add_rating_to_book(slots, rating_slots, graph, user, utterance_node_id)
        
             


class RestaurantIntentProcessor:
    
    def __init__(self):
        self.restaurant_slot_names = ['country', 'state', 'city', 'spatial_relation','served_dish','restaurant_type','restaurant_name','party_size_number','party_size_description','poi','sort','timeRange','cuisine','facility']
        self.restaurant_essential = {'resturant_loc' : ['country', 'state', 'city'], 'restaurant_poi': ['spatial_relation', 'poi'], 'timeRange': ['timeRange'], 'restaurant_party' : ['party_size_number','party_size_description']}
        self.now_equivalents = ['now', 'today']
    
    def create_restaurant_references(self, slots, graph, utterance_node_id, user_id, start_dialog):
        restaurant_find_last = {1:self.restaurant_last_loc, 3:self.restaurant_last_timeRange, 2:self.restaurant_last_party_size}
        # Costruisco un dizionario con tutti i possibili slot, in modo tale da controllare facilmente se gli slot sono riempiti
        restaurant_slots = Dialog.check_true_values(slots, self.restaurant_slot_names)     
        # Per prenotare un ristorante serve come minimo: city(oppure poi+spatial relation), timeRange, party_size_description oppure le loro versioni più specifiche
        conditions = self.restaurant_slot_is_filled(restaurant_slots)
        if not conditions:
            self.add_restaurant(restaurant_slots, graph, utterance_node_id, start_dialog)
        else:
            for cond in conditions:
                restaurant_find_last[cond](graph, utterance_node_id, restaurant_slots)
            self.add_restaurant(restaurant_slots, graph, utterance_node_id, start_dialog)

    def restaurant_slot_is_filled(self, restaurant_slots):
        res = []
        c1 = restaurant_slots['city'] or restaurant_slots['state'] or restaurant_slots['country'] or (restaurant_slots['poi'] and restaurant_slots['spatial_relation'])
        c2 = restaurant_slots['party_size_number'] or restaurant_slots['party_size_description']
        c3 = restaurant_slots['timeRange']                                                                                               
        if not c1:
            res.append(1)
        if not c2:
            res.append(2)
        if not c3:
            res.append(3)
        return res
    
    def find_last_party_size(self, node, graph):
        if graph.nodes[node]['labels'] == ':Dialog':
            return [-1, -1]
        next_node = None
        for neigh in graph.neighbors(node):
            if graph.nodes[neigh]['labels'] == ':PartySize':
                return [neigh, graph.nodes[neigh]['labels'].lower()[1:]]
            if graph[node][neigh]['label'] == 'FOLLOWS' and (graph.nodes[neigh]['labels'] == ':Utterance' or graph.nodes[neigh]['labels'] == ':Dialog'):
                next_node = neigh
        return self.find_last_party_size(next_node, graph)

    def restaurant_last_party_size(self, graph, utterance_node_id, restaurant_slots):
        last_size, size_type = self.find_last_party_size(utterance_node_id, graph)
        if last_size == -1:
            print('No party size or description previously mentioned')
        else:
            #print("Party size found: ", last_size)
            restaurant_slots[size_type] = last_size
            #add_party_size_to_graph(graph, restaurant_slots, utterance_node_id)

    def find_last_party_size(self, node, graph):
        if graph.nodes[node]['labels'] == ':Dialog':
            return [-1, -1]
        next_node = None
        for neigh in graph.neighbors(node):
            if graph.nodes[neigh]['labels'] == ':PartySize':
                return [neigh, graph.nodes[neigh]['labels'].lower()[1:]]
            if graph[node][neigh]['label'] == 'FOLLOWS' and (graph.nodes[neigh]['labels'] == ':Utterance' or graph.nodes[neigh]['labels'] == ':Dialog'):
                next_node = neigh
        return self.find_last_party_size(next_node, graph)

    def restaurant_last_loc(self, graph, utterance_node_id, restaurant_slots):
        last_loc, loc_type = self.find_last_location(utterance_node_id, graph)
        if last_loc == -1:
            print('No city or country or state or point of interest previously mentioned')
        else:
            #print("Location found: ", last_loc)
            restaurant_slots[loc_type] = last_loc
            #add_restaurant_location_to_graph(graph, restaurant_slots, utterance_node_id)

    def restaurant_last_timeRange(self, graph, utterance_node_id, restaurant_slots):
        last_time = self.find_last_timeRange(utterance_node_id, graph)
        if last_time == -1:
            return
            print('No time or day previously mentioned')
        else:
            #print("Time/day found: ", last_time)
            restaurant_slots['timeRange'] = last_time
            #add_restaurant_timeRange_to_graph(graph, restaurant_slots, utterance_node_id)

    def find_last_timeRange(self, node, graph):
        if graph.nodes[node]['labels'] == ':Dialog':
            return -1
        next_node = None
        for neigh in graph.neighbors(node):
            if graph.nodes[neigh]['labels'] ==':TimeRange':
                return neigh
            if graph[node][neigh]['label'] == 'FOLLOWS' and (graph.nodes[neigh]['labels'] == ':Utterance' or graph.nodes[neigh]['labels'] == ':Dialog'):
                next_node = neigh
        return self.find_last_timeRange(next_node, graph)

    def add_restaurant_location_to_graph(self, graph, restaurant_slots, utterance_node_id):
        ret = False
        #if not (restaurant_slots['poi'] and  restaurant_slots['spatial_relation']):
        if restaurant_slots['city']:
            ret = True
            graph.add_node(restaurant_slots['city'], labels=":City")
            graph.add_edge(utterance_node_id, restaurant_slots['city'], label='REFERS_TO')
            if restaurant_slots['state']:
                graph.add_node(restaurant_slots['state'], labels=":State") 
                graph.add_edge(restaurant_slots['city'], restaurant_slots['state'], label="IN")
                graph.add_edge(restaurant_slots['state'], restaurant_slots['city'], label="CONTAINS")
                if restaurant_slots['country']:
                    graph.add_node(restaurant_slots['country'], labels=":Country") 
                    graph.add_edge(restaurant_slots['state'], restaurant_slots['country'], label="IN")
                    graph.add_edge(restaurant_slots['country'], restaurant_slots['state'], label="CONTAINS")
                elif restaurant_slots['country']:
                    graph.add_node(restaurant_slots['country'], labels=":Country") 
                    graph.add_edge(restaurant_slots['city'], restaurant_slots['country'], label="IN")
                    graph.add_edge(restaurant_slots['country'], restaurant_slots['city'], label="CONTAINS")
        elif restaurant_slots['state']:
            ret = True
            graph.add_node(restaurant_slots['state'], labels=":State")
            graph.add_edge(utterance_node_id, restaurant_slots['state'], label='REFERS_TO')
            if restaurant_slots['country']:
                graph.add_node(restaurant_slots['country'], labels=":Country")
                graph.add_edge(restaurant_slots['state'], restaurant_slots['country'], label="IN")
                graph.add_edge(restaurant_slots['country'], restaurant_slots['state'], label="CONTAINS")
        elif restaurant_slots['country']:
            ret = True
            graph.add_node(restaurant_slots['country'], labels=":Country")
            graph.add_edge(utterance_node_id, restaurant_slots['country'], label='REFERS_TO')
        return ret

       
    def add_restaurant_poi_to_graph(self, graph, restaurant_slots, utterance_node_id):
        ret = False
        if restaurant_slots['poi'] and restaurant_slots['spatial_relation']:
            graph.add_node(restaurant_slots['poi'], labels=":Poi")
            graph.add_edge(utterance_node_id, restaurant_slots['poi'], label="REFERS_TO", spatial_relation=restaurant_slots['spatial_relation'])
            ret = True
        return ret
    
        
    def add_restaurant_timeRange_to_graph(self, graph, restaurant_slots, utterance_node_id, start_dialog):
        ret = True
        if restaurant_slots['timeRange']=='tomorrow':
            tomorrow = start_dialog + timedelta(days=1)
            graph.add_node(tomorrow.strftime("%d/%m/%Y"), labels=":Date")
            graph.add_edge(utterance_node_id, tomorrow.strftime("%d/%m/%Y"), label='REFERS_TO')
        elif (not restaurant_slots['timeRange']) or (restaurant_slots['timeRange'].lower() in self.now_equivalents):
            date_id = start_dialog.strftime("%d/%m/%Y")
            graph.add_node(date_id, labels=":Date")
            graph.add_edge(utterance_node_id, date_id, label='REFERS_TO')
        else:
            ret = False
        return ret
        '''
        if(restaurant_slots['timeRange']):
            graph.add_node(restaurant_slots['timeRange'], labels=":TimeRange")
            graph.add_edge(utterance_node_id, restaurant_slots['timeRange'], label="REFERS_TO")
        '''    
    
    def add_restaurant_cusine_to_graph(self, graph, restaurant_slots, utterance_node_id):
        if restaurant_slots['restaurant_name']:
            graph.add_node(restaurant_slots['restaurant_name'], labels=":Restaurant")
            graph.add_edge(utterance_node_id, restaurant_slots['restaurant_name'], label='REFERS_TO')
        if restaurant_slots['restaurant_type']:
            graph.add_node(restaurant_slots['restaurant_type'], labels=":Restaurant")
            graph.add_edge(utterance_node_id, restaurant_slots['restaurant_type'], label='REFERS_TO')
        if restaurant_slots['served_dish']:
            graph.add_node(restaurant_slots['served_dish'], labels=":Dish")
            graph.add_edge(utterance_node_id, restaurant_slots['served_dish'], label='REFERS_TO')
        if restaurant_slots['cuisine']:
            graph.add_node(restaurant_slots['cuisine'], labels=":Cusine")
            graph.add_edge(utterance_node_id, restaurant_slots['cusine'], label='SERVES')

    def add_party_size_to_graph(self, graph, restaurant_slots, utterance_node_id):
        global global_node_id
        ret = False
        if restaurant_slots['party_size_number']: #numero di persone
            ret = True
            graph.add_node(global_node_id, labels=":PartySize", size=restaurant_slots['party_size_number'])
            graph.add_edge(utterance_node_id, global_node_id, label="REFERS_TO")
            global_node_id += 1
        elif restaurant_slots['party_size_description']:
            ret = True
            graph.add_node(global_node_id, labels=":PartySize", desc=restaurant_slots['party_size_description'])
            graph.add_edge(utterance_node_id, global_node_id, label="REFERS_TO")
            global_node_id += 1
        return ret

    def print_reservation_message(self, restaurant_slots):
        slots = ['city', 'state', 'country']
        party_size = ""
        loc = ""
        for slot in slots:
            if restaurant_slots[slot] is not None:
                loc = loc + str(restaurant_slots[slot]) + " "
        if restaurant_slots['poi'] is not None and restaurant_slots['spatial_relation'] is not None:
                loc = loc + restaurant_slots['spatial_relation']+" "+restaurant_slots['poi']
        if restaurant_slots['party_size_number'] is not None:
            party_size = restaurant_slots['party_size_number']
        else:
             party_size = restaurant_slots['party_size_description']
        if restaurant_slots['timeRange'] is None:
            time = "now"
        else: 
            time = restaurant_slots['timeRange']
        print("Reserved for ", party_size, ", time: ",time, " in ", loc)

    def add_restaurant(self, restaurant_slots, graph, utterance_node_id, start_dialog):
        ret_string = "Insert "
        loc = self.add_restaurant_location_to_graph(graph, restaurant_slots, utterance_node_id)
        poi = self.add_restaurant_poi_to_graph(graph, restaurant_slots, utterance_node_id)
        time = self.add_restaurant_timeRange_to_graph(graph, restaurant_slots, utterance_node_id, start_dialog)
        party_size = self.add_party_size_to_graph(graph, restaurant_slots, utterance_node_id)
        cond = [loc or poi, party_size, time]
        messages = ["location or poi", "party size or description", "time or day"]
        for i in range(len(cond)):
            if not cond[i]:
                ret_string += messages[i] + ", "
        ret_string = ret_string[:-2]
        if not ((loc or poi) and party_size and time):
            print(ret_string) # Da stampare nella GUI
        else:
            self.print_reservation_message(restaurant_slots)
            #print("Reserved") # Selezionare risposta affermativa randomicamente da una lista
        self.add_restaurant_cusine_to_graph(graph, restaurant_slots, utterance_node_id)

    def find_last_location(self, node, graph):
        if graph.nodes[node]['labels'] == ':Dialog':
            return [-1, -1]
        next_node = None
        for neigh in graph.neighbors(node):
            if graph.nodes[neigh]['labels'] in [':City', ':Country', ':State', ':Poi']:
                return [neigh, graph.nodes[neigh]['labels'].lower()[1:]]
            # In un dialogo, per ogni utterance può esserci un solo arco follows
            if graph[node][neigh]['label'] == 'FOLLOWS' and (graph.nodes[neigh]['labels'] == ':Utterance' or graph.nodes[neigh]['labels'] == ':Dialog'):
                next_node = neigh
        return self.find_last_location(next_node, graph)


import requests as req

class WeatherIntentProcessor:
    def __init__(self):
        self.weather_slot_names = ['condition_description', 'condition_temperature', 'city', 'country', 'timeRange', 'state']
        self.now_equivalents = ['now', 'today']

    def find_last_location(self, node, graph):
        if graph.nodes[node]['labels'] == ':Dialog':
            return [-1, -1]
        next_node = None
        for neigh in graph.neighbors(node):
            if graph.nodes[neigh]['labels'] in [':City', ':Country', ':State', ':Poi']:
                return [neigh, graph.nodes[neigh]['labels'].lower()[1:]]
            # In un dialogo, per ogni utterance puó esserci un solo arco follows
            if graph[node][neigh]['label'] == 'FOLLOWS' and (graph.nodes[neigh]['labels'] == ':Utterance' or graph.nodes[neigh]['labels'] == ':Dialog'):
                next_node = neigh
        return self.find_last_location(next_node, graph)
        
    def extract_weather_info(self, city):
        url = 'https://wttr.in/{}'.format(city)
        url += '?format=%l%t%C'
        res = req.get(url)
        
        if ('Unknown location' in res.text):
            return -1
        info = list(res.text)
        if('+' in info):
            location = info[0:info.index('+')]
            temp = info[info.index('+'):info.index('°')] + list('°') + list(info[info.index('°') + 1])
        elif('-' in info):
            location = info[0:info.index('-')]
            temp = info[info.index('-'):info.index('°')] + list('°') + list(info[info.index('°') + 1])
        
        weather = info[info.index('°')+2:]
        location = ''.join([str(elem) for elem in location])
        temp = ''.join([str(elem) for elem in temp])
        weather = ''.join([str(elem) for elem in weather])
        
        return location, temp, weather

    def create_weather_references(self, slots, graph, utterance_node_id, user_id, start_dialog):
        weather_slots = Dialog.check_true_values(slots, self.weather_slot_names)     # Costruisco un dizionario con tutti i possibili slot, in modo tale da controllare facilmente se gli slot sono riempiti
        if weather_slots['city'] or weather_slots['country'] or weather_slots['state']:
            self.add_weather(weather_slots, graph, utterance_node_id, start_dialog)
        else:
            self.weather_last_loc(graph, utterance_node_id, weather_slots, start_dialog)

    def add_weather(self, weather_slots, graph, utterance_node_id, start_dialog):
        if weather_slots['city']:
            graph.add_node(weather_slots['city'], labels=":City")
            graph.add_edge(utterance_node_id, weather_slots['city'], label='REFERS_TO')
            location, temp, weather = self.extract_weather_info(weather_slots['city'])
            if not (-1 in [location, temp, weather]): 
                print(location, temp, weather)
            else:
                print("Unknown location")
            if weather_slots['state']:
                graph.add_node(weather_slots['state'], labels=":State") 
                graph.add_edge(weather_slots['city'], weather_slots['state'], label="IN")
                graph.add_edge(weather_slots['state'], weather_slots['city'], label="CONTAINS")  
                if weather_slots['country']:
                    graph.add_node(weather_slots['country'], labels=":Country") 
                    graph.add_edge(weather_slots['country'], weather_slots['state'], label="IN")
                    graph.add_edge(weather_slots['state'], weather_slots['country'], label="CONTAINS")
                    #graph.add_edge(utterance_node_id, weather_slots['country'], label='REFERS_TO')
            elif weather_slots['country']:     
                graph.add_node(weather_slots['country'], labels=":Country") 
                graph.add_edge(weather_slots['city'], weather_slots['country'], label="IN")
                graph.add_edge(weather_slots['country'], weather_slots['city'], label="CONTAINS")
        else:
            if weather_slots['state']:
                graph.add_node(weather_slots['state'], labels=":State")
                graph.add_edge(utterance_node_id, weather_slots['state'], label='REFERS_TO')
                if weather_slots['country']:
                    graph.add_node(weather_slots['country'], labels=":Country")
                    graph.add_edge(weather_slots['state'], weather_slots['country'], label="IN")
                    graph.add_edge(weather_slots['country'], weather_slots['state'], label="CONTAINS")
                    location, temp, weather = self.extract_weather_info(weather_slots['country'])
                    if not (-1 in [location, temp, weather]): 
                        print(location, temp, weather)
                    else:
                        print("Unknown location")
                else:
                    location, temp, weather = self.extract_weather_info(weather_slots['state'])
                    if not (-1 in [location, temp, weather]): 
                        print(location, temp, weather)
                    else:
                        print("Unknown location")
            elif weather_slots['country']:
                graph.add_node(weather_slots['country'], labels=":Country")
                graph.add_edge(utterance_node_id, weather_slots['country'], label='REFERS_TO')
                location, temp, weather = self.extract_weather_info(weather_slots['country'])
                if not (-1 in [location, temp, weather]): 
                    print(location, temp, weather)
                else:
                    print("Unknown location")

        if weather_slots['timeRange']=='tomorrow':
            tomorrow = start_dialog + timedelta(days=1)
            graph.add_node(tomorrow.strftime("%d/%m/%Y"), labels=":Date")
            graph.add_edge(utterance_node_id, tomorrow.strftime("%d/%m/%Y"), label='REFERS_TO') 
        if (not weather_slots['timeRange']) or (weather_slots['timeRange'].lower() in self.now_equivalents):
            date_id = start_dialog.strftime("%d/%m/%Y")
            graph.add_node(date_id, labels=":Date")
            graph.add_edge(utterance_node_id, date_id, label='REFERS_TO')

    def weather_last_loc(self, graph, utterance_node_id, weather_slots, start_dialog): #Se object_type=None prende semplicemente l'ultimo libro presente nel dialogo
        # Vedere sulla doc di networkx come navigare gli archi del grafo
        # Fondamentalmente va fatta una visita a ritroso a partire dalla utterance attuale fino a trovare
        # il primo libro di tipo object_tipe (se specificato, altrimenti il primo libro di qualsiasi tipo)
        last_loc, loc_type = self.find_last_location(utterance_node_id, graph)
        if last_loc == -1:
            print('Missing location, please specify a place') #TODO mettere nella gui
        else:
            #print("Location found: " + last_loc)
            weather_slots[loc_type] = last_loc
            self.add_weather(weather_slots, graph, utterance_node_id, start_dialog)