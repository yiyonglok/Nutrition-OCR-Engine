from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.bottomsheet import MDListBottomSheet
from kivymd.toast import toast
from kivymd.uix.list import IRightBodyTouch, OneLineAvatarIconListItem
from kivymd.uix.menu import MDDropdownMenu
import os
from kivymd.uix.snackbar import Snackbar
from kivy.metrics import dp
from kivymd.uix.filemanager import MDFileManager
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
import threading
from kivy.clock import mainthread
from functools import partial
from kivymd.uix.progressbar import MDProgressBar
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivymd.uix.dialog import MDDialog
import numpy
import skimage.io
from PIL import Image, ImageOps
import os
from sys import platform
from kivymd.uix.list import OneLineAvatarListItem
from kivy.properties import StringProperty
import matplotlib

import ImageProcessor
import numpy as np
import shutil
import KTrain
import KTester
import DataPrinter
import time
import random
import math
import MultiOutputMLP as mlp
import image_processor_8x8
import HeatMapBuilder
import LabelToLetterTranslation as l2l
import get_word_images as gwi

KV = '''

<Item>

    ImageLeftWidget:
        source: root.source


<Content>
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "120dp"

    MDTextField:
        hint_text: "File Name"
        
<Content2>
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "120dp"

    MDTextField:
        hint_text: "Centroid Count"
        
    MDTextField:
        hint_text: "Set Name"
        
<Content3>
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "120dp"
    
    MDLabel:
        text: "You must save model before training"

    MDTextField:
        hint_text: "File Name"
        
<Content4>
    orientation: "vertical"
    spacing: "6dp"
    size_hint_y: None
    height: "500dp"

    MDLabel:
        text: "                                        Multi MLP Results"
        theme_text_color: "Custom"
        size_hint_y : .2
        text_color: "white"
        font_size : '17dp'

    Image :
        id: heatmap_label
        source : 'UIassets/NoImage.png'
        size: self.texture_size
        
    MDLabel:
        text: "                                        Binary MLP Results"
        theme_text_color: "Custom"
        size_hint_y : .2
        text_color: "white"
        font_size : '17dp'
        
    Image :
        id: heatmap_label
        source : 'UIassets/NoImage.png'
        

        

MDBoxLayout:

    MDBottomNavigation:
        #panel_color: "#eeeaea"
        selected_color_background: "#03a9f4"
        text_color_active: "black"

        MDBottomNavigationItem:
            name: 'screen 1'
            text: 'Train'
            icon: 'UIassets/training_icon.png'


            MDTopAppBar:
                id: model_label
                pos_hint: {"top": 1}
                elevation: 4
                title: "No Model Loaded"
                left_action_items: [["menu", lambda x: app.callback(x)]]


            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .25, "center_y": .7}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :

                    orientation : 'vertical'
                    padding : '10dp'
                    Image : 
                        id: letter_image
                        source : 'UIassets/NoImage.png'


                    MDLabel :
                        text : '   Letter Data'
                        size_hint_y : .2
                        font_style : 'H6'
                    
                    
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: letter_label
                        text : '    No Data Selected' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'



                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3


                        MDTextButton:
                            text: "   Inspect         "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_letter_inspect_alert_dialog()
                            disabled: False


                        MDTextButton:
                            id: letter_new_set_label
                            text: "New Set         "
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.file_manager_open()
                            disabled: False



                        MDTextButton:
                            id: letter_select_set_label
                            text: "Select Set      "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_letter_list_bottom_sheet()
                            disabled: False





            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .75, "center_y": .70}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :
                    orientation : 'vertical'
                    padding : '10dp'
                    Image :
                        id: non_letter_image
                        source : 'UIassets/NoImage.png'



                    MDLabel :
                        text : '   Non-Letter Data'
                        size_hint_y : .2
                        font_style : 'H6'
                        
                            
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: nonletter_label
                        text : '    No Data Selected' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'

                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3


                        MDTextButton:
                            
                            text: "   Inspect         "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_nonletter_inspect_alert_dialog()
                            disabled: False


                        MDTextButton:
                            id: nonletter_new_set_label
                            text: "New Set         "
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.nonletter_file_manager_open()
                            disabled: False

                        MDTextButton:
                            id: nonletter_select_set_label
                            text: "Select Set      "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_nonletter_list_bottom_sheet()
                            disabled: False


            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .25, "center_y": .25}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :
                    orientation : 'vertical'
                    padding : '10dp'
                    Image :
                        id: feature_image 
                        source : 'UIassets/NoImage.png'



                    MDLabel :
                        text : '    Feature Set'
                        size_hint_y : .2
                        font_style : 'H6'
                        
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: feature_label
                        text : '    No Features Trained' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'



                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3


                        MDTextButton:
                            text: "     Inspect         "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_feature_inspect_alert_dialog()



                        MDTextButton:
                            id: feature_train_label
                            text: "Train         "
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.check_for_model()
                            disabled: False


                        MDTextButton:
                            text: "Select Set        "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_Feature_list_bottom_sheet()


            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .75, "center_y": .25}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :
                    id: model_image
                    orientation : 'vertical'
                    padding : '10dp'
                    Image : 
                        source : 'UIassets/NoImage.png'



                    MDLabel :
                        text : '    Model Weights'
                        size_hint_y : .2
                        font_style : 'H6'
                    
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: modelweights_label
                        text : '    Training Incomplete' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'


                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3


                        MDTextButton:
                            text: "           Train         "
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_model_popup()


                        MDTextButton:
                            text: "      View Results         "
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.DiagramPopup()


                            










        MDBottomNavigationItem:
            name: 'screen 2'
            text: 'Test'
            icon: 'UIassets/testing_icon.png'

            MDTopAppBar:
                id: model_label_testing
                pos_hint: {"top": 1}
                elevation: 4
                title: "No Model Loaded"
                left_action_items: [["menu", lambda x: app.callback(x)]]


            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .25, "center_y": .7}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :

                    orientation : 'vertical'
                    padding : '10dp'
                    Image : 
                        id: testing_image
                        source : 'UIassets/NoImage.png'


                    MDLabel :
                        text : '   Testing Image'
                        size_hint_y : .2
                        font_style : 'H6'
                    
                    
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: image_label
                        text : '    No Image Selected' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'



                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3


                        MDTextButton:
                            id: letter_new_set_label
                            text: "                              Import"
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.image_file_manager_open()
                            disabled: False


            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .75, "center_y": .70}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :
                    orientation : 'vertical'
                    padding : '10dp'
                    Image :
                        id: processed_image
                        source : 'UIassets/NoImage.png'



                    MDLabel :
                        text : '   Processed Image'
                        size_hint_y : .2
                        font_style : 'H6'
                        
                            
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: processed_label
                        text : '    No Image Processed' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'

                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3


                        MDTextButton:
                            
                            text: "                              Process"
                            theme_text_color: "Custom"
                            size_hint_y : .2
                            text_color: "blue"
                            font_size : '17dp'
                            on_press: app.show_imageprocess_popup()
                            disabled: False

                    



            MDCard:
                size_hint: .45, .40
                focus_behavior: True
                pos_hint: {"center_x": .5, "center_y": .25}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2

                MDBoxLayout :
                    orientation : 'vertical'
                    padding : '10dp'
                    Image :
                        id: heatmap_label
                        source : 'UIassets/NoImage.png'



                    MDLabel :
                        text : '   Detection & Recognition'
                        size_hint_y : .2
                        font_style : 'H6'
                        
                    MDSeparator:
                        height: "1dp"
                    
                    MDLabel :
                        id: testing_label
                        text : '    Test Incomplete' 
                        size_hint_y : .2
                        font_style : 'Subtitle1'



                    MDSeparator:
                        height: "1dp"

                    MDGridLayout:
                        size_hint_y: 0.2
                        cols: 3





                        MDTextButton:
                            id: feature_train_label
                            text: "   Run Detection         "
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.show_detectprocess_popup()
                            disabled: False
                            
                        MDTextButton:
                            id: feature_train_label
                            text: "   View Results         "
                            theme_text_color: "Custom"
                            text_color: "blue"
                            size_hint_y : .2
                            font_size : '17dp'
                            on_press: app.show_recognition_list_bottom_sheet()
                            disabled: False

        MDBottomNavigationItem:
            name: 'screen 3'
            text: 'About'
            icon: 'UIassets/about_icon.png'
            

   
            MDCard:
                size_hint: .45, .40
                pos_hint: {"center_x": .5, "center_y": .5}
                md_bg_color: "white"
                unfocus_color: "white"
                focus_color: "#e3f2fd"
                elevation: 2
                
                
                MDBoxLayout :
                    orientation : 'vertical'
                    padding : '10dp'
                                            
                    Image : 
                        source : 'UIassets/nutrimaclogo.png'
                        
                    Image : 
                        source : 'UIassets/maclogo.png'
                
                    MDGridLayout:
                        size_hint_y: 0.3
                        rows: 2
                                     
                        MDLabel:
                            text: 'Build: v1.0'
                            halign: 'center'    
                
                                
                        MDLabel:
                            text: 'Contact: Nurtimac@mcmaster.ca'
                            halign: 'center'    

'''


class Content(BoxLayout):
    pass

class Content2(BoxLayout):
    pass

class Content3(BoxLayout):
    pass

class Content4(BoxLayout):
    pass


class ListItem(OneLineAvatarIconListItem):
    pass

class Item(OneLineAvatarListItem):
    divider = None
    source = StringProperty()


class Test(MDApp):
    dialog = None
    model_name = ""
    holder_filepath = ""
    current_model = ""
    current_letter_data = ""
    current_non_letter_data = ""
    nonletter_filename = False
    untrained_change = False
    current_feature_set = ""
    lock_data = False
    done_thread = False
    non_letter_progress = False
    letter_data_length = ""
    letter_classes = ""
    nonletter_data_length = ""
    nonletter_classes = ""
    current_centroid_count = ""
    image_filename = False
    current_test_image = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager, select_path=self.select_path
        )

    def DataLock(self):
        pass

    def DataUnlock(self):
        self.root.ids.letter_new_set_label.disabled = False
        self.root.ids.letter_select_set_label.disabled = False
        self.root.ids.nonletter_select_set_label.disabled = False
        self.root.ids.nonletter_new_set_label.disabled = False

    def DiagramPopup(self):
        popup = Popup(title='Training Results',
                      content=Content4(),
                       size_hint=(None, None), size=(1000, 1200))

        popup.open()

    def show_popup(self,inst):
        self.progress_bar = MDProgressBar()
        self.popup = Popup(
            title='Splicing Data...',
            content=self.progress_bar,
            auto_dismiss=False,  # dialog does NOT close if click outside it
            size_hint=(None, None),
            size=(800, 400)
        )
        self.popup.bind(on_open=lambda x: self.run_thread())
        # self.progress_bar.max = 100
        self.progress_bar.value = 10

        self.popup.open()
        print(self.progress_bar.value)


    @mainthread
    def update_progress_bar(self, val, _):
        self.progress_bar.value = val


    def run_thread(self):

        text_item = self.dialog.content_cls.children[0].text
        destination = "LetterData/Images/" + text_item + ".png"
        DataPrinter.printData(self.holder_filepath,destination)
        self.root.ids.letter_image.source = destination
        t1 = threading.Thread(target=self.process_some_data)
        t1.start()

    def process_some_data(self):

        text_item = self.dialog.content_cls.children[0].text
        self.dialog.dismiss()
        self.ProcessLetterData(self.holder_filepath, text_item)
        self.popup.dismiss()

    def show_detectprocess_popup(self):
        self.progress_bar = MDProgressBar()
        self.popup = Popup(
            title='Processing..',
            content=self.progress_bar,
            auto_dismiss=False,  # dialog does NOT close if click outside it
            size_hint=(None, None),
            size=(800, 400)
        )
        self.popup.bind(on_open=lambda x: self.run_detectprocess_thread())
        # self.progress_bar.max = 100
        self.progress_bar.value = 10

        self.popup.open()
        print(self.progress_bar.value)

    @mainthread
    def update_detectprocess_progress_bar(self, val, _):
        self.progress_bar.value = val

        if self.progress_bar.value == 100:
            self.root.ids.heatmap_label.source = 'myheatmap.png'

    def run_detectprocess_thread(self):

        t1 = threading.Thread(target=self.process_some_detectprocess_data)
        t1.start()

    def process_some_detectprocess_data(self):

        with open('bo_hidden_weights.npy', 'rb') as opened_file:
            bo_hidden_weights = np.load(opened_file)
        with open('bo_output_weights.npy', 'rb') as opened_file:
            bo_output_weights = np.load(opened_file)
        with open('mo_hidden_weights.npy', 'rb') as opened_file:
            mo_hidden_weights = np.load(opened_file)
        with open('mo_output_weights.npy', 'rb') as opened_file:
            mo_output_weights = np.load(opened_file)
        with open('centroid_data_150centroids_alex_shrunk_lowercase.npy', 'rb') as opened_file:
            centroid_data = np.load(opened_file)


        self.CharacterDetection(bo_hidden_weights,bo_output_weights,mo_hidden_weights,mo_output_weights,centroid_data)
        self.popup.dismiss()


    def show_imageprocess_popup(self):
        self.progress_bar = MDProgressBar()
        self.popup = Popup(
            title='Processing..',
            content=self.progress_bar,
            auto_dismiss=False,  # dialog does NOT close if click outside it
            size_hint=(None, None),
            size=(800, 400)
        )
        self.popup.bind(on_open=lambda x: self.run_imageprocess_thread())
        # self.progress_bar.max = 100
        self.progress_bar.value = 10

        self.popup.open()
        print(self.progress_bar.value)

    @mainthread
    def update_imageprocess_progress_bar(self, val, _):
        self.progress_bar.value = val

        if self.progress_bar.value == 100:
            self.root.ids.processed_image.source = 'label.jpg'
            self.root.ids.processed_label.text = "Image Processing Complete"

    def run_imageprocess_thread(self):

        t1 = threading.Thread(target=self.process_some_imageprocess_data)
        t1.start()

    offset = 0
    image_data = np.array(1)
    image_width = 0
    image_height = 0
    img_pixel_data = np.array(1)
    x = 0
    y = 0

    def process_some_imageprocess_data(self):
        with open('bo_hidden_weights.npy', 'rb') as opened_file:
            bo_hidden_weights = np.load(opened_file)
        with open('bo_output_weights.npy', 'rb') as opened_file:
            bo_output_weights = np.load(opened_file)
        with open('mo_hidden_weights.npy', 'rb') as opened_file:
            mo_hidden_weights = np.load(opened_file)
        with open('mo_output_weights.npy', 'rb') as opened_file:
            mo_output_weights = np.load(opened_file)
        with open('centroid_data_150centroids_alex_shrunk_lowercase.npy', 'rb') as opened_file:
            centroid_data = np.load(opened_file)

        self.offset, self.image_data, self.image_width, self.image_height, self.img_pixel_data, self.x, self.y = self.ProcessTestImage(bo_hidden_weights,bo_output_weights,mo_hidden_weights,mo_output_weights,centroid_data)
        self.popup.dismiss()


    def show_model_popup(self):
        self.progress_bar = MDProgressBar()
        self.popup = Popup(
            title='Splicing Data...',
            content=self.progress_bar,
            auto_dismiss=False,  # dialog does NOT close if click outside it
            size_hint=(None, None),
            size=(800, 400)
        )
        self.popup.bind(on_open=lambda x: self.run_model_thread())
        # self.progress_bar.max = 100
        self.progress_bar.value = 10

        self.popup.open()
        print(self.progress_bar.value)

    def run_model_thread(self):


        t1 = threading.Thread(target=self.process_some_model_data)
        t1.start()

    def process_some_model_data(self):

        self.TrainWeights()
        self.popup.dismiss()

    @mainthread
    def update_model_progress_bar(self, val, _):
        self.progress_bar.value = val

        if self.progress_bar.value == 100:
            self.root.ids.model_image.source = 'UIassets/accuracy.png'





    def show_nonletter_popup(self,inst):
        self.progress_bar = MDProgressBar()
        self.popup = Popup(
            title='Splicing Data...',
            content=self.progress_bar,
            auto_dismiss=False,  # dialog does NOT close if click outside it
            size_hint=(None, None),
            size=(800, 400)
        )
        self.popup.bind(on_open=lambda x: self.run_nonletter_thread())
        # self.progress_bar.max = 100
        self.progress_bar.value = 10

        self.popup.open()
        print(self.progress_bar.value)


    @mainthread
    def update_nonletter_progress_bar(self, val, _):
        self.progress_bar.value = val


    def run_nonletter_thread(self):

        text_item = self.dialog.content_cls.children[0].text
        destination = "NonLetterData/Images/" + text_item +".png"
        DataPrinter.nonLetterCollage(self.holder_filepath, destination)
        self.root.ids.non_letter_image.source = destination
        # self.ProcessNonLetterData(self.holder_filepath + "/", text_item)
        # destination = "LetterData/Images/" + text_item + ".png"
        # DataPrinter.printData(self.holder_filepath,destination)
        # self.root.ids.letter_image.source = destination
        t1 = threading.Thread(target=self.process_some_nonletter_data)
        t1.start()

    def process_some_nonletter_data(self):

        text_item = self.dialog.content_cls.children[0].text
        self.dialog.dismiss()
        self.ProcessNonLetterData(self.holder_filepath + "/", text_item)
        self.popup.dismiss()

    def show_feature_popup(self,inst):
        self.progress_bar = MDProgressBar()
        self.popup = Popup(
            title='Generating Features...',
            content=self.progress_bar,
            auto_dismiss=False,  # dialog does NOT close if click outside it
            size_hint=(None, None),
            size=(800, 400)
        )
        self.popup.bind(on_open=lambda x: self.run_feature_thread())
        # self.progress_bar.max = 100
        self.progress_bar.value = 10

        self.popup.open()
        print(self.progress_bar.value)


    @mainthread
    def update_feature_progress_bar(self, val, _):
        self.progress_bar.value = val

        if self.progress_bar.value == 100:

            filename = self.dialog.content_cls.children[0].text
            centroid_count = int(self.dialog.content_cls.children[1].text)
            centroid_data = "Models/" + self.current_model + '/FeatureData/' + self.current_feature_set
            centroid_image_destinaion = "Models/" + self.current_model + '/FeatureData/' + filename + "/" + "centroid_image.png"
            # DataPrinter.centroidCollage(centroid_data, centroid_count, centroid_image_destinaion)
            # DataPrinter.image_resize(centroid_image_destinaion)

            self.root.ids.feature_image.source = 'UIassets/CentroidImages/collage_150.jpg'



    def run_feature_thread(self):


        t1 = threading.Thread(target=self.process_some_feature_data)
        t1.start()


    def process_some_feature_data(self):

        self.dialog.dismiss()
        self.TrainFeatures()
        self.popup.dismiss()


    def file_manager_open(self):
        self.file_manager.show(os.path.expanduser("~"))  # output manager to the screen
        self.manager_open = True

    def nonletter_file_manager_open(self):
        self.file_manager.show(os.path.expanduser("~"))  # output manager to the screen
        self.manager_open = True
        self.nonletter_filename = True
        print(self.nonletter_filename)

    def image_file_manager_open(self):
        self.file_manager.show(os.path.expanduser("~"))  # output manager to the screen
        self.manager_open = True
        self.image_filename = True

    def select_path(self, path: str):
        '''
        It will be called when you click on the file name
        or the catalog selection button.

        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        self.holder_filepath = path

        if self.nonletter_filename:
            self.show_nonletterset_alert_dialog()
            self.nonletter_filename = False

        elif self.image_filename:
            self.current_test_image = path
            self.root.ids.image_label.text = path[17:]
            src_path = path
            dst_path = "Models/" + self.current_model + '/Tests' + '/Image/'
            shutil.copy(src_path, dst_path)
            self.root.ids.testing_image.source = path
            self.image_filename = False
        else:
            self.show_letterset_alert_dialog()

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    def show_letter_inspect_alert_dialog(self):

        self.dialog = MDDialog(
            title="Letter Data Information",
            type="simple",
            items=[
                Item(text=""+ " pieces of data", source="UIassets/about_icon.png"),
                Item(text="" + " unique classe(s)", source="UIassets/about_icon.png"),
            ],
            buttons=[
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press=self.closeDialog
                ),
            ],
        )
        self.dialog.open()

    def show_nonletter_inspect_alert_dialog(self):

        self.dialog = MDDialog(
            title="Letter Data Information",
            type="simple",
            items=[
                Item(text="" + " pieces of data", source="UIassets/about_icon.png"),
                Item(text="" + " unique classe(s)", source="UIassets/about_icon.png"),
            ],
            buttons=[
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press=self.closeDialog
                ),
            ],
        )
        self.dialog.open()

    def show_feature_inspect_alert_dialog(self):

        self.dialog = MDDialog(
            title="Letter Data Information",
            type="simple",
            items=[
                Item(text=self.current_centroid_count + " centroids generated", source="UIassets/about_icon.png"),
            ],
            buttons=[
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press=self.closeDialog
                ),
            ],
        )
        self.dialog.open()

    def show_view_model_alert_dialog(self):

        self.dialog = MDDialog(
            title="Letter Data Information",
            type="simple",
            items=[
                Item(text=self.current_centroid_count + " centroids generated", source="UIassets/about_icon.png"),
            ],
            buttons=[
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press=self.closeDialog
                ),
            ],
        )
        self.dialog.open()

    def show_image_alert_dialog(self):

        self.dialog = MDDialog(
            text="New Letter Dataset Name",
            type="custom",
            content_cls=Content(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog
                ),
                MDFlatButton(
                    text="SUBMIT",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.show_nonletter_popup
                ),
            ],
        )
        self.dialog.open()


    def show_nonletterset_alert_dialog(self):

        self.dialog = MDDialog(
            text="New Letter Dataset Name",
            type="custom",
            content_cls=Content(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog
                ),
                MDFlatButton(
                    text="SUBMIT",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.show_nonletter_popup
                ),
            ],
        )
        self.dialog.open()



    def show_letterset_alert_dialog(self):
        self.dialog = MDDialog(
            text="New Letter Dataset Name",
            type="custom",
            content_cls=Content(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog
                ),
                MDFlatButton(
                    text="SUBMIT",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.show_popup
                ),
            ],
        )
        self.dialog.open()


    def show_alert_dialog(self):
        self.dialog = MDDialog(
            text="New Model Name",
            type="custom",
            content_cls=Content(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog
                ),
                MDFlatButton(
                    text="SUBMIT",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.grabText
                ),
            ],
        )
        self.dialog.open()

    def check_for_model(self):
        if self.current_model == "":
            self.show_save_model_alert_dialog()
        else:
            self.show_FeatureTraining_alert_dialog()

    def show_save_model_alert_dialog(self):
        self.dialog = MDDialog(
            text="New Model Name",
            type="custom",
            content_cls=Content3(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog
                ),
                MDFlatButton(
                    text="SUBMIT",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.grabText
                ),
            ],
        )
        self.dialog.open()


    def show_FeatureTraining_alert_dialog(self):
        self.dialog = MDDialog(
            text="New Model Name",
            type="custom",
            content_cls=Content2(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog
                ),
                MDFlatButton(
                    text="SUBMIT",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.show_feature_popup
                ),
            ],
        )
        self.dialog.open()

    def LoadFeatureImage(self):

        self.root.ids.feature_image.source = 'UIassets/CentroidImages/collage_150.jpg'

    def TrainFeatures(self):

        filename = self.dialog.content_cls.children[0].text
        centroid_count = int(self.dialog.content_cls.children[1].text)
        centroid_count = 5
        self.current_centroid_count = str(centroid_count)
        letter_data = "LetterData/" + self.current_letter_data

        self.DataLock()
        os.mkdir("Models/" + self.current_model + '/FeatureData/' + filename)
        label_folder = "Models/" + self.current_model + '/FeatureData/' + filename + "/" + "centroid_letters"
        centroid_folder = "Models/" + self.current_model + '/FeatureData/' + filename
        self.RunKTrain(filename,centroid_count,letter_data,label_folder,centroid_folder)
        self.current_feature_set = filename + ".npy"

        non_letter_label_folder = "Models/" + self.current_model + '/FeatureData/' + filename + "/" + "centroid_non_letters"
        non_letter_data = "NonLetterData/" + self.current_non_letter_data
        centroid_data = "Models/" + self.current_model + '/FeatureData/' + self.current_feature_set
        self.TestLabelData(non_letter_data, centroid_data, non_letter_label_folder)
        self.root.ids.feature_label.text = "    " + self.current_feature_set
        Clock.schedule_once(partial(self.update_feature_progress_bar, 100), 0)

        # centroid_image_destinaion = "Models/" + self.current_model + '/FeatureData/' + filename + "/" + "centroid_image.png"
        # DataPrinter.centroidCollage(centroid_data, centroid_count, centroid_image_destinaion)
        # DataPrinter.image_resize(centroid_image_destinaion)
        #
        # self.root.ids.feature_image.source = centroid_image_destinaion

        self.dialog.dismiss()

    def grabLetterText(self, inst):
        for obj in self.dialog.content_cls.children:
            if isinstance(obj, MDTextField):
                self.ProcessLetterData(self.holder_filepath, obj.text)
                destination = "LetterData/Images/" + obj.text + ".png"
                DataPrinter.printData(self.holder_filepath,destination)
                self.root.ids.letter_image.source = destination

        self.dialog.dismiss()



    def grabNonLetterText(self, inst):
        # for obj in self.dialog.content_cls.children:
        #     if isinstance(obj, MDTextField):
        #         print(self.holder_filepath)
        #         self.ProcessNonLetterData(self.holder_filepath + "/", obj.text)
        # self.dialog.dismiss()
        self.non_letter_progress = True
        self.show_popup

    def grabText(self, inst):
        for obj in self.dialog.content_cls.children:
            if isinstance(obj, MDTextField):
                self.NewFolder(obj.text)
        self.dialog.dismiss()

    def NewFolder(self, model_name):
        os.mkdir("Models/" + model_name)
        os.mkdir("Models/" + model_name + '/LetterData')
        os.mkdir("Models/" + model_name + '/NonLetterData')
        os.mkdir("Models/" + model_name + '/FeatureData')
        os.mkdir("Models/" + model_name + '/FeatureData' + '/Images')
        os.mkdir("Models/" + model_name + '/ModelWeights')
        os.mkdir("Models/" + model_name + '/Tests')
        os.mkdir("Models/" + model_name + '/Tests' + '/Image')
        os.mkdir("Models/" + model_name + '/Tests' + '/ProcessedImage')
        os.mkdir("Models/" + model_name + '/Tests' + '/Heatmap')
        os.mkdir("Models/" + model_name + '/Tests' + '/Predictions')

        self.model_name = model_name
        self.current_model = model_name
        self.root.ids.model_label.title = model_name
        self.root.ids.model_label_testing.title = model_name

        print(self.current_model)
        if self.current_letter_data != "" or self.current_non_letter_data != "":
            self.save_model()
        else:
            self.root.ids.letter_label.text = "    No Data Selected"
            self.root.ids.nonletter_label.text = "    No Data Selected"

    def show_recognition_list_bottom_sheet(self):
        bottom_sheet_menu = MDListBottomSheet()
        Data = np.genfromtxt("", dtype=str,
                             encoding=None, delimiter=":")
        count = 0
        for i in Data:
            bottom_sheet_menu.add_item(
                f"{i[1]}",
                lambda x, y=i: self.callback_for_openmodelmenu_items(
                    f"{y}"
                ),icon = str(i[0])
            )
            count += 1
        bottom_sheet_menu.open()



    def closeDialog(self, inst):
        self.dialog.dismiss()

    def ModelList(self):
        folder = './Models'
        sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        return sub_folders

    def show_list_bottom_sheet(self):
        bottom_sheet_menu = MDListBottomSheet()
        models = self.ModelList()
        for i in models:
            bottom_sheet_menu.add_item(
                f"{i}",
                lambda x, y=i: self.callback_for_openmodelmenu_items(
                    f"{y}"
                ),
            )
        bottom_sheet_menu.open()

    def LetterDataList(self):
        folder = './LetterData'
        sub_folders = [entry for entry in os.listdir(folder) if os.path.isfile(os.path.join(folder, entry))]
        return sub_folders

    def NonLetterDataList(self):
        folder = './NonLetterData'
        sub_folders = [entry for entry in os.listdir(folder) if os.path.isfile(os.path.join(folder, entry))]
        return sub_folders

    def FeatureSetDataList(self):
        folder = './Models/' + self.current_model + '/FeatureData'
        sub_folders = [entry for entry in os.listdir(folder) if os.path.isfile(os.path.join(folder, entry))]
        return sub_folders

    def show_letter_list_bottom_sheet(self):
        bottom_sheet_menu = MDListBottomSheet()
        data = self.LetterDataList()
        for i in data:
            bottom_sheet_menu.add_item(
                f"{i}",
                lambda x, y=i: self.callback_for_menu_items(
                    f"{y}"
                ),
            )
        bottom_sheet_menu.open()

    def show_nonletter_list_bottom_sheet(self):
        bottom_sheet_menu = MDListBottomSheet()
        data = self.NonLetterDataList()
        for i in data:
            bottom_sheet_menu.add_item(
                f"{i}",
                lambda x, y=i: self.callback_for_nonlettermenu_items(
                    f"{y}"
                ),
            )
        bottom_sheet_menu.open()

    def show_Feature_list_bottom_sheet(self):

        if self.current_model == "":
            print("no model")
            self.show_no_model_alert_dialog()
        else:
            bottom_sheet_menu = MDListBottomSheet()
            data = self.FeatureSetDataList()
            for i in data:
                bottom_sheet_menu.add_item(
                    f"{i}",
                    lambda x, y=i: self.callback_for_Feature_menu_items(
                        f"{y}"
                    ),
                )
            bottom_sheet_menu.open()

    def show_no_model_alert_dialog(self):
        self.dialog = MDDialog(
            text="To select a feature set, open saved model or create a new one.",
            buttons=[

                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.closeDialog

                ),
            ],
        )
        self.dialog.open()

    def callback_for_Feature_menu_items(self, *args):
        self.root.ids.feature_label.text = "    " + args[0]
        self.current_feature_set = args[0]
        self.root.ids.feature_image.source = "UIassets/CentroidImages/collage_150.jpg"
        toast(args[0])

    def callback_for_menu_items(self, *args):
        self.root.ids.letter_label.text = "    " + args[0]
        self.current_letter_data = args[0]
        self.root.ids.letter_image.source = "LetterData/Images/" + args[0][:-4] + ".png"
        self.check_train()
        toast(args[0])

    def callback_for_openmodelmenu_items(self, *args):
        self.root.ids.model_label.title = args[0]
        self.current_model = args[0]
        toast(args[0])
        self.load_model()

    def callback_for_nonlettermenu_items(self, *args):
        self.root.ids.nonletter_label.text = "    " + args[0]
        self.current_non_letter_data = args[0]
        self.root.ids.non_letter_image.source = "NonLetterData/Images/" + args[0][:-4] + ".png"
        self.check_train()
        toast(args[0])

    def ProcessLetterData(self,filepath,filename):

        data_set = self.read_letter_images(filepath,filename)
        # np.save("LetterData/" + filename, data_set)
        # self.root.ids.letter_label.text = "    " + filename + ".npy"
        # self.current_letter_data = filename + ".npy"
        # self.check_train()
        # destination = "LetterData/Images/" + filename + ".png"
        # DataPrinter.printData(self.holder_filepath, destination)
        # self.root.ids.letter_image.source = destination


    def ProcessNonLetterData(self,filepath,filename):

        for path in os.listdir(filepath):
            # check if current path is a file
            if os.path.isfile(os.path.join(filepath, path)):

                if path != ".DS_Store":
                    ImageProcessor.image_resize(filepath + path)

        non_letter_data_newest = ImageProcessor.read_nonletter_images(filepath)
        self.nonletter_data_length = str(len(non_letter_data_newest) // 16)
        self.nonletter_classes = "1"
        save_location = "NonLetterData/" + filename
        np.save(save_location, non_letter_data_newest)
        self.root.ids.nonletter_label.text = filename + ".npy"
        self.current_non_letter_data = filename + ".npy"
        # destination = "NonLetterData/Images/" + filename +".png"
        # DataPrinter.nonLetterCollage(filepath, destination)
        # self.root.ids.non_letter_image.source = destination
        self.check_train()




    def build(self):
        self.theme_cls.material_style = "M3"
        self.theme_cls.theme_style = "Light"
        menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": f"{i}",
                "height": dp(56),
                "on_release": lambda x=f"{i}": self.menu_callback(x),
             } for i in ["New Model", "Open Model", "Save Model"]
        ]
        self.menu = MDDropdownMenu(
            items=menu_items,
            width_mult=4,
        )

        return Builder.load_string(KV)

    def callback(self, button):
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self, text_item):
        self.menu.dismiss()
        if text_item == "Open Model":
            self.show_list_bottom_sheet()
        elif text_item == "New Model":
            self.show_alert_dialog()
        elif text_item == "Save Model":
            Snackbar(text=text_item).open()
            self.save_model()

    def save_model(self):

        if self.current_model == "":
            self.show_alert_dialog()
        else:

            if self.current_letter_data != "":
                original_letterdata = r"LetterData/" + self.current_letter_data
                target_letterdata = r'Models/' + self.current_model + '/LetterData/' + self.current_letter_data
                shutil.rmtree('Models/' + self.current_model + '/LetterData')
                os.mkdir("Models/" + self.current_model + '/LetterData')
                shutil.copyfile(original_letterdata, target_letterdata)

            if self.current_non_letter_data != "":
                original_nonletterdata = r"NonLetterData/" + self.current_non_letter_data
                target_nonletterdata = r'Models/' + self.current_model + '/NonLetterData/' + self.current_non_letter_data
                shutil.rmtree('Models/' + self.current_model + '/NonLetterData')
                os.mkdir("Models/" + self.current_model + '/NonLetterData')
                shutil.copyfile(original_nonletterdata, target_nonletterdata)

            self.check_train()
            self.check_lock()
            self.check_image()

    def load_model(self):

        self.root.ids.letter_label.text = (os.listdir('Models/' + self.current_model + '/LetterData/'))[0]
        self.current_letter_data = (os.listdir('Models/' + self.current_model + '/LetterData/'))[0]
        self.root.ids.nonletter_label.text = (os.listdir('Models/' + self.current_model + '/NonLetterData/'))[0]
        self.current_non_letter_data = (os.listdir('Models/' + self.current_model + '/NonLetterData/'))[0]

        self.check_train()
        self.check_lock()
        self.check_image()


    def check_train(self):

        if self.current_letter_data != "" and self.current_non_letter_data != "":
            self.root.ids.feature_train_label.disabled = False

    def check_lock(self):

        directory = "Models/" + self.current_model + "/FeatureData"

        if os.listdir(directory):
            self.DataLock()
        else:
            self.DataUnlock()

    def check_image(self):

        letter_directory = "LetterData/Images/" + self.current_letter_data[:-4] + ".png"
        non_letter_directory = "NonLetterData/Images/" + self.current_non_letter_data[:-4] + ".png"

        if os.path.exists(letter_directory):
            self.root.ids.letter_image.source = letter_directory

        if os.path.exists(non_letter_directory):
            self.root.ids.non_letter_image.source = non_letter_directory


    def get_image_paths(self,path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file or '.jpg' in file:
                    files.append(os.path.join(r, file))
        return files

    def image_name_parser(self,filename):

        if platform == 'darwin':
            splitval = "/"
        else:
            splitval = "\\"

        file_path = filename.split(splitval)
        print(file_path)
        return int(file_path[-1][3:6])

    def read_letter_images(self,filepath,filename):
        load_letters = self.get_image_paths(filepath)
        print(filepath)
        # load_letters = load_letters[:10]
        print(len(load_letters))

        data_array = []
        progress = 0
        for file in load_letters:
            print(f"Progress: {progress * 100 / len(load_letters)}%")
            Clock.schedule_once(partial(self.update_progress_bar, progress * 100 / len(load_letters)), 0)
            if filepath != "non_letters":
                classifier = self.image_name_parser(file)
            else:
                classifier = 0

            # print(classifier)
            letter = skimage.io.imread(file, as_gray=True)
            if len(data_array) == 0:
                data_array = numpy.array(self.sampler(letter))
            else:
                data_array = numpy.concatenate((data_array, self.sampler(letter)), axis=0)
            progress += 1

        data_array = numpy.array(data_array)
        # print(data_array[-1])
        self.letter_data_length = str(len(data_array) // 16)
        self.letter_classes = str(len((np.unique(data_array[:,-1]))))

        numpy.save("LetterData/" + filename, data_array)
        self.root.ids.letter_label.text = "    " + filename + ".npy"
        self.current_letter_data = filename + ".npy"
        self.check_train()
        # self.printData(self.holder_filepath, destination)
        #self.root.ids.letter_image.source = destination

    def sampler(self, image, classifier):
        image_sample = []
        temp_array = []
        for column in range(0, len(image[0]), 8):
            for row in range(0, len(image[0]), 8):
                image_sample.append(image[column:column + 8, row:row + 8])
                temp_array.append(numpy.concatenate(
                    (numpy.array([num for sublist in image_sample[-1] for num in sublist]), [classifier])))
        return temp_array

    def NormalizeData(self,data_array):

        data_array = data_array.astype(float)

        for i in range(len(data_array)):
            mean = np.mean(data_array[i][:64])
            sd = np.std(data_array[i][:64])
            data_array[i][:64] -= mean
            if sd > 0:
                data_array[i][:64] /= sd

        return data_array

    def SquishData(self,data_array):

        return (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))

    def GenerateCentroids(self, CENTROID_COUNT):

        lower_bound = 0
        upper_bound = 255
        centroids = np.random.randint(lower_bound, upper_bound, size=(CENTROID_COUNT, 64))

        return centroids

    def CentroidDistance(self,data_array, centroids):

        distance = np.zeros((len(data_array), len(centroids)))

        for i in range(len(data_array)):
            for j in range(len(distance[i])):
                distance[i][j] = np.linalg.norm(data_array[i][:64] - centroids[j])

        return distance

    def LabelData(self,distance):

        label = np.zeros(len(distance))

        return (label + np.argmin(distance, axis=1))

    def CentroidMean(self,label, data_array, centroids):

        new_centroids = np.zeros((len(centroids), 64))

        for i in range(len(label)):
            new_centroids[int(label[i])] += data_array[i][:64]

        unique, counts = np.unique(label, return_counts=True)
        label_count = dict(zip(unique, counts))

        for key in label_count:
            new_centroids[int(key)] = new_centroids[int(key)] / label_count[key]

        for x in range(len(new_centroids)):
            if not np.any(new_centroids[x]):
                new_centroids[x] = centroids[x]

        return new_centroids

    def CentroidDifference(self,new_centroid, centroids):

        difference = np.zeros(len(centroids))

        for i in range(len(centroids)):
            difference[i] = np.linalg.norm(new_centroid[i] - centroids[i])

        return difference

    def FeatureMapping(self,x, c, label):

        out = np.zeros(len(x))
        for i in range(len(x)):
            z = np.linalg.norm(x[i][:64] - c[int(label[i])])
            mean = np.mean(x[i][:64])
            norm = mean - z
            out[i] = max(0.0, (norm))

        return out

    def calculate_z(self,letter_array_i, neuron_weights):
        dot_product = -np.dot(letter_array_i, neuron_weights)
        z_value = 1 / (1 + np.exp(dot_product))
        return z_value

    def RunKTrain(self,filename, count, sourcename, label_folder, centroid_folder):

        with open(sourcename, 'rb') as opened_file:
            data_array = np.load(opened_file)

        # shuffle data
        indexArray = list(range(np.size(data_array, 0)))
        data_array = data_array[indexArray]
        # data_array = data_array[29456:29456+12600]

        SAMPLE_SIZE = len(data_array)
        CENTROID_COUNT = count

        # Squish Data
        # data_array = SquishData(data_array)

        # Normalize
        # data_array = NormalizeData(data_array)

        # Kmeans
        centroids = self.GenerateCentroids(CENTROID_COUNT)
        previous_centroids = centroids
        difference = np.zeros(CENTROID_COUNT)
        difference.fill(200)
        counter = 0
        initial_difference = 0

        while np.max(difference) > 50:
            distance = self.CentroidDistance(data_array, centroids)
            label = self.LabelData(distance)
            new_centroids = self.CentroidMean(label, data_array, centroids)
            difference = self.CentroidDifference(new_centroids, centroids)
            centroids = new_centroids
            print(difference)
            print(np.max(difference))
            counter += 1
            if counter == 1:
                initial_difference = np.max(difference)
            Clock.schedule_once(partial(self.update_feature_progress_bar, (1-(np.max(difference)/initial_difference))*100), 0)
            print(1 - (np.max(difference) / initial_difference))


        # Feature Mapping
        feature_map = self.FeatureMapping(data_array, centroids, label)

        # Centroid Labelling
        new_array = data_array[:len(label)]
        new_array = np.concatenate((new_array, np.c_[label]), axis=1)

        np.save(centroid_folder, centroids)
        np.save(label_folder, new_array)

    def TestCentroidDistance(self,data_array, centroids):
        distance = np.zeros((len(data_array), len(centroids)))
        label = np.zeros(len(distance))

        for i in range(len(data_array)):
            if (i % (len(data_array) / 10)) < 1:
                print(f"Progress: {math.floor(i * 100 / len(data_array))}%")
            for j in range(len(distance[i])):
                distance[i][j] = np.linalg.norm(data_array[i][:64] - centroids[j])

        return label + np.argmin(distance, axis=1)

    def TestLabelData(self,data_file, centroid_file, file_name):
        print("Labelling:", data_file)
        with open(centroid_file, 'rb') as opened_file:
            centroids = np.load(opened_file)

        with open(data_file, 'rb') as opened_file:
            data = np.load(opened_file)

        label = self.TestCentroidDistance(data, centroids)
        label_data = data[:len(label)]
        label_data = np.concatenate((label_data, np.c_[label]), axis=1)

        np.save(file_name, label_data)

        return label_data

    def TrainWeights(self):
        self.TrainMOMLP()
        Clock.schedule_once(partial(self.update_model_progress_bar, 50), 0)
        self.TrainBOMLP()
        Clock.schedule_once(partial(self.update_model_progress_bar, 100), 0)
        self.root.ids.modelweights_label.text = 'Training Complete'

    def TrainMOMLP(self):
        alpha = 0.0001
        hidden_neurons = 1000
        EPOCHS = 10
        binary_output = False
        directory_path = "150Centroids"
        letter_data_path = f"{directory_path}/data_array_150centroids.npy"
        non_letter_data_path = f"{directory_path}/nonletter_data_array_150centroids.npy"
        centroid_file_path = f"{directory_path}/centroid_data_150centroids.npy"

        X = mlp.load_training_data(letter_data_path, non_letter_data_path, centroid_file_path, binary_output)
        X = mlp.shuffle_data(X)
        # X = mlp.shuffle_data(mlp.generate_random_data(10000))
        # print(len(X))
        # print(X[0])

        X_classifiers, X = mlp.separate_label(X)
        # print("X_classifier collection:", collections.Counter(X_classifiers))

        output_neurons = len(set(X_classifiers))

        hidden_weights, output_weights, X_classifiers_vectors = mlp.initialize_neural_network(hidden_neurons,
                                                                                              output_neurons,
                                                                                              X,
                                                                                              X_classifiers)

        # MLP Training
        start = time.time()  # Start Timer
        total_error_list, hidden_weights, output_weights = mlp.train_model(EPOCHS,
                                                                           X, X_classifiers, X_classifiers_vectors,
                                                                           alpha, hidden_weights, output_weights)

        mlp.save_weights(directory_path, output_neurons, hidden_weights, output_weights)

        end = time.time()
        print("Training Time:", end - start)
        time_string = "Training Time: " + str(end - start)

        matplotlib.pyplot.plot(total_error_list)
        matplotlib.pyplot.title(label=f"Alpha: {alpha}, Epochs: {EPOCHS}")
        matplotlib.pyplot.savefig("plots/multi.png")

    def TrainBOMLP(self):
        alpha = 0.0001
        hidden_neurons = 1000
        EPOCHS = 10
        binary_output = True
        directory_path = "150centroids"
        letter_data_path = f"{directory_path}/data_array_150centroids.npy"
        non_letter_data_path = f"{directory_path}/nonletter_data_array_150centroids.npy"
        centroid_file_path = f"{directory_path}/centroid_data_150centroids.npy"

        X = mlp.load_training_data(letter_data_path, non_letter_data_path, centroid_file_path, binary=binary_output)
        X = mlp.shuffle_data(X)
        # X = mlp.shuffle_data(mlp.generate_random_data(10000))
        # print(len(X))
        # print(X[0])

        X_classifiers, X = mlp.separate_label(X)
        # print("X_classifier collection:", collections.Counter(X_classifiers))

        output_neurons = len(set(X_classifiers))
        # print(output_neurons)

        hidden_weights, output_weights, X_classifiers_vectors = mlp.initialize_neural_network(hidden_neurons,
                                                                                              output_neurons,
                                                                                              X,
                                                                                              X_classifiers)

        # MLP Training
        start = time.time()  # Start Timer
        total_error_list, hidden_weights, output_weights = mlp.train_model(EPOCHS,
                                           X, X_classifiers, X_classifiers_vectors,
                                           alpha, hidden_weights, output_weights)
        end = time.time()
        print("Training Time:", end - start)
        time_string = "Training Time: " + str(end - start)

        mlp.save_weights(directory_path, output_neurons, hidden_weights, output_weights)

        matplotlib.pyplot.plot(total_error_list)
        matplotlib.pyplot.title(label=f"Alpha: {alpha}, Epochs: {EPOCHS}")
        matplotlib.pyplot.savefig("plots/binary.png")

    def ProcessTestImage(self,bo_hidden_weights, bo_output_weights, mo_hidden_weights, mo_output_weights, centroid_data):
        centroid_folder = '150centroids_alex_shrunk_lowercase'
        bo_hidden_weights_path = f'{centroid_folder}/bo_hidden_weights.npy'
        bo_output_weights_path = f'{centroid_folder}/bo_output_weights.npy'
        mo_hidden_weights_path = f'{centroid_folder}/mo_hidden_weights.npy'
        mo_output_weights_path = f'{centroid_folder}/mo_output_weights.npy'
        centroid_file_path = f'{centroid_folder}/centroid_data_150centroids_alex_shrunk_lowercase.npy'

        start = time.time()  # Start Timer

        offset, image_data, image_width, image_height, img_pixel_data, x, y = self.single_image_processor(
            image_path="images_to_process/nutrition-label.jpg", save_file=False, crop_image=False,
            remove_bad_samples=False, resize_by_height=False)
        return offset, image_data, image_width, image_height, img_pixel_data, x, y

    VALID_FORMATS = (".jpeg", ".jpg", ".png")
    SAMPLE_SIZE = 32
    MAX_RESOLUTION = (1080, 1080)
    REDUCTION_FACTOR = 4

    def resize_image(self,img):
        width, height = img.size
        max_width, max_height = self.MAX_RESOLUTION
        if (width > max_width) or (height > max_height):
            img = img.resize((width // self.REDUCTION_FACTOR, height // self.REDUCTION_FACTOR))
        return img

    def resize_image_by_height(self,img):

        letter_height_endpoints = []
        img_pixel_data = img.load()
        width, height = img.size

        # boost contrast
        # for i_index in range(height):
        #    for j_index in range(width):
        #        if img_pixel_data[j_index, i_index] > 110:
        #            img_pixel_data[j_index, i_index] = 255
        #        else:
        #            img_pixel_data[j_index, i_index] = 0

        sample_32x32 = []
        # build 32x32 sample
        for y_index in range(0, height):
            temp_array = []
            for x_index in range(0, width):
                temp_array.append(img_pixel_data[x_index, y_index])
            sample_32x32.append(temp_array)

        img_pixel_data = np.array(sample_32x32)

        # find top and bottom of letter
        is_space = False
        for i in range(int(height / 2), -1, -1):
            if not is_space and np.count_nonzero(img_pixel_data[i, :] == 0) < 10:
                is_space = True
                letter_height_endpoints.append(i)
            if is_space:
                break
        if not is_space:
            letter_height_endpoints.append(0)

        is_space = False
        for i in range(int(height / 2), height):
            if not is_space and np.count_nonzero(img_pixel_data[i, :] == 0) < 10:
                is_space = True
                letter_height_endpoints.append(i)
            if is_space:
                break
        if not is_space:
            letter_height_endpoints.append(height)

        letter_size = letter_height_endpoints[1] - letter_height_endpoints[0]
        print("size:", letter_size)

        if letter_size > 25:
            aspect_ratio = width / height
            height_reduction = int(height - (letter_size - 25))
            if height_reduction < 33:
                height_reduction = 33
            width_reduction = int(aspect_ratio * height_reduction)
            img = img.resize((width_reduction, height_reduction))
            print(height_reduction)

        return img

    def determine_max_valid_index(self,sample_size, offset, dimension_len):
        indices = [i for i in range(0, dimension_len, offset)]
        i = -1
        while (indices[i] + sample_size) > dimension_len - 1:
            i -= 1
        return indices[i]

    def save_letter(self, image_matrix, name):
        image = Image.fromarray(image_matrix)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        file_path = f"{name}.jpg"
        image.save(file_path)

    def sampler(self, image):
        image_sample = []
        temp_array = []
        for column in range(0, len(image[0]), 8):
            for row in range(0, len(image[0]), 8):
                image_sample.append(image[column:column + 8, row:row + 8])
                temp_array.append(np.array([num for sublist in image_sample[-1] for num in sublist]))
        return temp_array

    def is_quality_sample(self, image):
        if np.min(image) > 200:
            return False
        min_first_col = np.min(image[:, 0])
        min_last_col = np.min(image[:, -1])
        min_first_row = np.min(image[0, :])
        min_last_row = np.min(image[-1, :])
        if (min_first_col < 5) != (
                min_last_col < 5):  # and (np.count_nonzero(image[:, 0] == min_first_col) > 5 or np.count_nonzero(image[:, 0] == min_last_col) > 5):
            return False
        if (min_first_row < 5) != (
                min_last_row < 5):  # and (np.count_nonzero(image[:, 0] == min_first_row) > 5 or np.count_nonzero(image[:, 0] == min_last_row) > 5):
            return False
        return True

    def dynamically_crop_image(self, image):
        image = np.array(image)
        width = len(image)
        b = np.min(image, axis=0)

        letter_width_endpoints = []
        letter_height_endpoints = []

        # crops sides
        is_space = False
        for i in range(int(width / 2), -1, -1):
            if not is_space and b[i] > 5:
                is_space = True
                image[:, :i] = 255
                letter_width_endpoints.append(i)
            if is_space:
                break
        # if not is_space:
        #    letter_width_endpoints.append(0)

        is_space = False
        for i in range(int(width / 2), width):
            if not is_space and b[i] > 5:
                is_space = True
                image[:, i:] = 255
                letter_width_endpoints.append(i)
            if is_space:
                break
        # if not is_space:
        #    letter_width_endpoints.append(32)

        d = np.min(image, axis=1)

        # crop top and bottom
        is_space = False
        for i in range(int(width / 2), -1, -1):
            if not is_space and d[i] > 5:
                is_space = True
                image[:i, :] = 255
                letter_height_endpoints.append(i)
            if is_space:
                break
        # if not is_space:
        #    letter_height_endpoints.append(0)

        is_space = False
        for i in range(int(width / 2), width):
            if not is_space and d[i] > 5:
                is_space = True
                image[i:, :] = 255
                letter_height_endpoints.append(i)
            if is_space:
                break
        # if not is_space:
        #    letter_height_endpoints.append(32)

        if len(letter_height_endpoints) == 2:
            shift = letter_height_endpoints[0] - (width - letter_height_endpoints[1])
            # if abs(shift) > 15:
            #    None
            if shift < -1:
                shift = abs(shift)
                new_column = np.zeros((int(shift / 2), width)) + 255
                image = image[:-int(shift / 2), :]
                image = np.concatenate((new_column, image), axis=0)

            elif shift > 1:
                new_column = np.zeros((int(shift / 2), width)) + 255
                image = image[int(shift / 2):, :]
                image = np.concatenate((image, new_column), axis=0)

        if len(letter_width_endpoints) == 2:
            shift = letter_width_endpoints[0] - (width - letter_width_endpoints[1])
            # if abs(shift) > 15:
            #    None
            if shift < -1:
                shift = abs(shift)
                new_column = np.zeros((width, int(shift / 2))) + 255
                image = image[:, :-int(shift / 2)]
                image = np.concatenate((new_column, image), axis=1)

            elif shift > 1:
                new_column = np.zeros((width, int(shift / 2))) + 255
                image = image[:, int(shift / 2):]
                image = np.concatenate((image, new_column), axis=1)

        # print(letter_width_endpoints)
        # print("width of letter?", letter_width_endpoints[1] - letter_width_endpoints[0])
        # print("height of letter?", letter_height_endpoints[1] - letter_height_endpoints[0])
        # print("image shape now: ", np.shape(image))

        return image

    def rebuild_32x32(self, letter_data, sample_index):
        letter_data = np.reshape(letter_data, (16, 64))

        temp_array = []

        for i in range(len(letter_data)):
            temp_array.append(np.reshape(letter_data[i], (8, 8)))

        temp_image = np.block([[temp_array[0], temp_array[1], temp_array[2], temp_array[3]],
                               [temp_array[4], temp_array[5], temp_array[6], temp_array[7]],
                               [temp_array[8], temp_array[9], temp_array[10], temp_array[11]],
                               [temp_array[12], temp_array[13], temp_array[14], temp_array[15]]])

        # temp_image = numpy.reshape(letter_data[0], (8, 8))
        print(temp_image)
        self.save_letter(temp_image, "ImageTestingOuput/image" + str(sample_index))

    def multi_image_processor(self, offset=4, save_image=False):
        # image paths list
        image_paths = []
        file_names = []

        # walk 'images_to_process' folder to find images to work on
        for (root, dirs, files) in os.walk('images_to_process'):
            for file in files:
                if file.endswith(self.VALID_FORMATS):
                    image_paths.append(os.path.join(root, file))
                    file_names.append(file.split(".")[0])

        # loop through found images/paths
        for image_count, image_path in enumerate(image_paths):
            image_data_8x8 = []
            with Image.open(image_path) as img:
                # get image dimensions, grayscale, load pixel data
                width, height = img.size
                img = ImageOps.grayscale(img)
                img = self.resize_image(img)
                img_pixel_data = img.load()
                # boost contrast
                for i_index in range(height):
                    for j_index in range(width):
                        if img_pixel_data[j_index, i_index] > 110:
                            img_pixel_data[j_index, i_index] = 255
                        else:
                            img_pixel_data[j_index, i_index] = 0

            if save_image:
                # get file name
                file_name = image_path.split(".")[0]
                file_name = file_name.split("\\")[1]
                img.save(f"{file_name}.jpg")

            # determine max index to take sample within image dimensions
            max_width_index = self.determine_max_valid_index(self.SAMPLE_SIZE, offset, width)
            max_height_index = self.determine_max_valid_index(self.SAMPLE_SIZE, offset, height)

            # take 32x32 sample in 8x8 pieces for each offset step
            for i in range(0, max_height_index + 1, offset):
                print(f"{file_names[image_count]} : row {i} of {max_height_index}\r", end="")
                for j in range(0, max_width_index + 1, offset):
                    sample_32x32 = []
                    # build 32x32 sample
                    for y_index in range(i, i + self.SAMPLE_SIZE):
                        temp_array = []
                        for x_index in range(j, j + self.SAMPLE_SIZE):
                            temp_array.append(img_pixel_data[x_index, y_index])
                        sample_32x32.append(temp_array)
                    # cut sample into 16 8x8 pieces and add to 8x8 image data
                    pieces_8x8 = self.sampler(np.array(sample_32x32))
                    for piece in pieces_8x8:
                        image_data_8x8.append(piece)

            # convert to numpy array and save
            image_data_8x8 = np.array(image_data_8x8)
            print(f"\n{file_names[image_count]} : {image_data_8x8.shape}")
            np.save(f"{file_names[image_count]}_8x8_data", image_data_8x8)

    def single_image_processor(self, offset=4, image_path=None, save_file=False, crop_image=False, remove_bad_samples=False, resize_by_height=False):
        # validation
        if not image_path:
            print("Image not specified")
            exit()

        if not image_path.endswith(self.VALID_FORMATS):
            print("Image not valid format")
            exit()

        # get file name
        file_name = image_path.split(".")[0]

        # process image
        image_data_8x8 = []
        with Image.open(image_path) as img:
            # get image dimensions, grayscale, load pixel data
            img = self.resize_image(img)
            width, height = img.size
            img = ImageOps.grayscale(img)
            if resize_by_height:
                img = self.resize_image_by_height(img)
                width, height = img.size
            img_pixel_data = img.load()
            # boost contrast
            for i_index in range(height):
                for j_index in range(width):
                    if img_pixel_data[j_index, i_index] > 110:
                        img_pixel_data[j_index, i_index] = 255
                    else:
                        img_pixel_data[j_index, i_index] = 0

        # determine max index to take sample within image dimensions
        max_width_index = self.determine_max_valid_index(self.SAMPLE_SIZE, offset, width)
        max_height_index = self.determine_max_valid_index(self.SAMPLE_SIZE, offset, height)

        # take 32x32 sample in 8x8 pieces for each offset step
        count = 0
        for i in range(0, max_height_index + 1, offset):
            Clock.schedule_once(partial(self.update_progress_bar, (i/max_height_index)*100), 0)
            print(f"{file_name} : row {i} of {max_height_index}\r", end="")
            for j in range(0, max_width_index + 1, offset):
                sample_32x32 = []
                # build 32x32 sample
                for y_index in range(i, i + self.SAMPLE_SIZE):
                    temp_array = []
                    for x_index in range(j, j + self.SAMPLE_SIZE):
                        temp_array.append(img_pixel_data[x_index, y_index])
                    sample_32x32.append(temp_array)
                if crop_image:
                    sample_32x32 = self.dynamically_crop_image(sample_32x32)
                    # sample_32x32 = dynamically_crop_image(sample_32x32)
                else:
                    sample_32x32 = np.array(sample_32x32)
                if remove_bad_samples:
                    if not self.is_quality_sample(sample_32x32):
                        sample_32x32[:, :] = 255
                # save_letter(sample_32x32, f"reshaped_images/{count}")
                count += 1
                # cut sample into 16 8x8 pieces and add to 8x8 image data
                pieces_8x8 = self.sampler(np.array(sample_32x32))
                for piece in pieces_8x8:
                    image_data_8x8.append(piece)
        Clock.schedule_once(partial(self.update_imageprocess_progress_bar, 100), 0)
        # convert to numpy array
        image_data_8x8 = np.array(image_data_8x8)
        print(f"\n{file_name} : {image_data_8x8.shape}")

        if save_file:
            np.save(f"myfile", image_data_8x8)

        print("image data 8x8 shape: ", np.shape(image_data_8x8))
        return offset, image_data_8x8, width, height, img_pixel_data, max_height_index, max_width_index


    def CharacterDetection(self, bo_hidden_weights,bo_output_weights,mo_hidden_weights,mo_output_weights,centroid_data):

        centroid_folder = '150centroids_alex_shrunk_lowercase'
        centroid_file_path ='150centroids_alex_shrunk_lowercase/centroid_data_150centroids_alex_shrunk_lowercase.npy'
        image_array = self.HeatLabelData(self.image_data, centroid_file_path,  f"{centroid_folder}/Calories_data_150centroids_alex")

        X = mlp.load_testing_data(image_array, centroid_file_path)
        Clock.schedule_once(partial(self.update_detectprocess_progress_bar, 20), 0)

        heatMap = HeatMapBuilder.HeatMap(self.image_width, self.image_height, len(X), self.offset, self.img_pixel_data)
        heatMap.print_dimensions()
        # self.root.ids.heatmap_label.source = 'myheatmap.png'

        Clock.schedule_once(partial(self.update_detectprocess_progress_bar, 50), 0)

        # Test with MLP
        bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
        hits = numpy.nonzero(bo_predictions)
        for hit in hits[0]:
            heatMap.update_heat_map(hit)

        heatMap.print_heat_map(superimpose=False)
        Clock.schedule_once(partial(self.update_detectprocess_progress_bar, 60), 0)

        gwi.get_word_images(nutrition_label_image="label.jpg", heatmap_image="heatmap.jpg")
        Clock.schedule_once(partial(self.update_detectprocess_progress_bar, 90), 0)

        load_word_images = gwi.get_word_image_paths("words")
        counter = 0
        letter_recognitions_text_file = open("LetterRecognitions.txt", "w")
        binary_filtered_text_file = open("LetterRecognitionsBinaryFiltered.txt", "w")
        translated_text_file = open("LetterRecognitionsTranslatedFiltered.txt", "w")
        predicted_word_text_file = open("PredictedWords.txt", "w")
        Clock.schedule_once(partial(self.update_detectprocess_progress_bar, 100), 0)


        wordlist = []
        for file in load_word_images:

            offset, image_data, image_width, image_height, img_pixel_data, max_height_index, max_width_index = self.single_image_processor( \
                image_path=f"{file}", save_file=False, crop_image=True, \
                resize_by_height=True, remove_bad_samples=True)

            if len(image_data) > 0:

                image_array = self.HeatLabelData(image_data, \
                                           centroid_file_path, \
                                           f"{centroid_folder}/{counter}Calories_data_150centroids_alex")

                X = mlp.load_testing_data(image_array, centroid_file_path)

                bo_predictions = mlp.run_model(X, bo_hidden_weights, bo_output_weights)
                mo_predictions = mlp.run_model(X, mo_hidden_weights, mo_output_weights)
                hits = numpy.nonzero(bo_predictions)

                cleaned_mo_predictions, bo_predictions = l2l.remove_stacked_samples(mo_predictions, bo_predictions,
                                                                                    max_height_index, max_width_index)
                hits = numpy.nonzero(bo_predictions)

                # after cleanup
                empty_array = numpy.zeros(len(bo_predictions))
                empty_array[hits] = cleaned_mo_predictions[hits]

                letter_recognitions_text_file.write(f"{file}: {str(mo_predictions)}")
                letter_recognitions_text_file.write("\n")
                binary_filtered_text_file.write(f"{file}: {str(bo_predictions)}")
                binary_filtered_text_file.write("\n")

                if len(empty_array[hits]) > 0:
                    letter_translation_array = numpy.vectorize(l2l.labels_as_letters.get)(empty_array[hits])
                    translated_text_file.write(f"{file}: {str(letter_translation_array)}")
                    translated_text_file.write("\n")

                    word = l2l.translate_letters_to_words(letter_translation_array)
                    predicted_word_text_file.write(f"{file}: {word}")
                    predicted_word_text_file.write("\n")
                    print(word)
                    wordlist.append(word)

                counter += 1

        letter_recognitions_text_file.close()
        binary_filtered_text_file.close()
        translated_text_file.close()
        predicted_word_text_file.close()
        Clock.schedule_once(partial(self.update_progress_bar, 100, 0))
        numpy.save("PredictedWords", wordlist)




    def HeatCentroidDistance(self,data_array, centroids):
        distance = np.zeros((len(data_array), len(centroids)))
        label = np.zeros(len(distance))

        for i in range(len(data_array)):
            if (i % (len(data_array) / 10)) < 1:
                print(f"Progress: {math.floor(i * 100 / len(data_array))}%")
            for j in range(len(distance[i])):
                distance[i][j] = np.linalg.norm(data_array[i][:64] - centroids[j])

        return label + np.argmin(distance, axis=1)

    def HeatLabelData(self, data_file, centroid_file, file_name):
        print("Labelling:", data_file)
        with open(centroid_file, 'rb') as opened_file:
            centroids = np.load(opened_file)

        if isinstance(data_file, str):
            with open(data_file, 'rb') as opened_file:
                data = np.load(opened_file)
        else:
            data = data_file

        label = self.HeatCentroidDistance(data, centroids)
        label_data = data[:len(label)]
        label_data = np.concatenate((label_data, np.c_[label]), axis=1)

        np.save(file_name, label_data)

        return label_data

















