import unittest
import streamlit as st
# from streamlit.script_runner import RerunException

from national_flag_app import model, transform

class TestNationalFlagApp(unittest.TestCase):
    def test_model_loading(self):
        self.assertIsNotNone(model)
    
    def test_transform(self):
        self.assertIsNotNone(transform)
    
    def test_file_uploader(self):
        try:
            st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        except RerunException:
            pass

if __name__ == '__main__':
    unittest.main()
