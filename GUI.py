import networkx as nx
import ipywidgets as widgets
from ipywidgets import Button, Layout, jslink, IntText, IntSlider, Text, Output
import librosa
from IPython.display import display, Audio, clear_output
import librosa.display
import pysndfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from ipywidgets import TwoByTwoLayout, AppLayout, GridspecLayout
from IPython.display import Javascript
from google.colab import output
from base64 import b64decode
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class GUI:
    def __init__(self, controller):
        self.save_button = widgets.Button(description='Export Graph',
                                            disabled=False,
                                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                            tooltip='Print graph and exports it ',
                                            layout=Layout(height='auto', width='auto'),
                                            icon='check' # (FontAwesome names without the `fa-` prefix)
                                            )
        self.box =  TwoByTwoLayout(top_left=self.save_button,
                    bottom_left=None,
                    bottom_right=None,
                    top_right=None,
                    justify_items='center',
                    width="100%",
                    align_items='center',
                    align_content='center'
                    )
        
        self.slider = widgets.IntSlider(
                                        value=5,
                                        min=1,
                                        max=20,
                                        step=1,
                                        description='Duration (s):',
                                        disabled=False,
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='d'
                                    )
        self.controller = controller
        self.audio = "audio.wav"
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.out = widgets.Output(layout={'border': '1px solid black', 'width' : '90%'})
        self.record_button = widgets.Button(description='Record',
                                            disabled=False,
                                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                            tooltip='Click me',
                                            layout=Layout(height='auto', width='60%'),
                                            icon='' # (FontAwesome names without the `fa-` prefix)
                                            )
        self.transcription_text = widgets.Text( value='',
                                                placeholder='Transcription here',
                                                description='Transcription:',
                                                style = {'description_width':'initial'},
                                                layout=Layout(height='auto',  width='90%',margin='4px', padding_right="20px"),
                                                icon=''
                                                disabled=False
                                            )

        self.send_button = widgets.Button(
            description='Send',
            info='',
            layout=Layout(height='auto', width='60%', margin='4px'),
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )

        self.app = TwoByTwoLayout(top_left=self.record_button,
                    bottom_left=self.send_button,
                    bottom_right=self.out,
                    top_right=self.transcription_text,
                    justify_items='center',
                    width="80%",
                    align_items='center',
                    align_content='center'
                    )

        self.RECORD = """
                    const sleep = time => new Promise(resolve => setTimeout(resolve, time))
                    const b2text = blob => new Promise(resolve => {
                    const reader = new FileReader()
                    reader.onloadend = e => resolve(e.srcElement.result)
                    reader.readAsDataURL(blob)
                    })
                    var record = time => new Promise(async resolve => {
                    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
                    recorder = new MediaRecorder(stream)
                    chunks = []
                    recorder.ondataavailable = e => chunks.push(e.data)
                    recorder.start()
                    await sleep(time)
                    recorder.onstop = async ()=>{
                        blob = new Blob(chunks)
                        text = await b2text(blob)
                        resolve(text)
                    }
                    recorder.stop()
                    })
                    """   


    def send_button_handler(self,button=None):
        sentence = self.transcription_text.value
        if len(sentence)<1:
            return
        with self.out:
            clear_output()
            self.controller.interact(sentence)
            #qua chiama le funzionio

    
    def transcribe_speech(self,button=None):
        progress = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            description='Transcribing:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            style={'bar_color': 'green'},
            orientation='horizontal'
        )
        with self.out:
            display(progress)
            progress.value=2
        input_values = self.processor(self.speech, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
        # retrieve logits
        progress.value=3
        logits = self.model(input_values).logits
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        progress.value=5
        transcription = self.processor.batch_decode(predicted_ids)
        progress.value=8
        self.transcription_text.value=transcription[0].lower()
        progress.value=10
        with self.out:
            clear_output()
            print("Please, edit the transcription if there are any errors")
            #self.listen_to_audio()



    def listen_to_audio(self,button=None):
        self.speech, rate = librosa.load(self.audio, sr=16000)
        pysndfile.sndio.write('audio_ds.wav', self.speech, rate=rate, format='wav', enc='pcm16')

        with self.out:
            clear_output()
            #display(Audio(self.speech, rate=rate))

    def record(self,button=None):
        with self.out:
            clear_output()
            #self.out.append_stdout("Speak")
            print("Speak for {} seconds".format((self.slider.value)))
        filename = "audio.wav"
        display(Javascript(self.RECORD))
        s = output.eval_js('record(%d)' % (self.slider.value * 1000))
        b = b64decode(s.split(',')[1])
        with open(filename, 'wb+') as f:
            f.write(b)
        with self.out:
            #self.out.append_stdout("Done!")
            print("Done!")
        self.listen_to_audio()
        self.transcribe_speech()

    def export_graph(self, button=None):
        path = "example.graphml"
        with self.out:
            clear_output()
            print("Exporting grap at ", path)
            nx.write_graphml(self.controller.graph, path, named_key_ids=True)
            clear_output()
            print("Exported.")


    def initGUI(self):
        self.record_button.on_click(self.record)
        self.send_button.on_click(self.send_button_handler)
        self.save_button.on_click(self.export_graph)
        display(self.slider)
        display(self.app)
        display(self.box)
