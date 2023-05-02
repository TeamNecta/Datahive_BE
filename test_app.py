import unittest
import json
from app import app

class TestApp(unittest.TestCase):
    
    def test_home(self):
        with app.test_client() as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
    
    def test_upload(self):
        with app.test_client() as client:
            with open('test.csv', 'rb') as f:
                data = {'file': (f, 'test.csv')}
                response = client.post('/upload', data=data, content_type='multipart/form-data')
                self.assertEqual(response.status_code, 200)
                response_data = json.loads(response.data.decode('utf-8'))
                self.assertIn('data', response_data)
                self.assertIn('dataType', response_data)
                self.assertIn('cols', response_data)
                self.assertIn('columns', response_data)
                self.assertIn('table', response_data)
                self.assertIn('filename', response_data)
                self.assertIn('datatype_table', response_data)
    
    def test_advance_cleaning(self):
        with app.test_client() as client:
            with open('test.csv', 'rb') as f:
                data = {'file': (f, 'test.csv')}
                response_upload = client.post('/upload', data=data, content_type='multipart/form-data')
                response_data = json.loads(response_upload.data.decode('utf-8'))['data']
                data_dict = json.loads(response_data)
                response_cleaning = client.post('/advance_cleaning', data={'data': response_data, 'action': 'replace_missing', 'replace_column': 'B', 'replace_method': 'mean'})
                response_data = json.loads(response_cleaning.data.decode('utf-8'))
                self.assertIn('data', response_data)
                self.assertIn('dataType', response_data)
                self.assertIn('cols', response_data)
                self.assertIn('columns', response_data)
                self.assertIn('table', response_data)
                self.assertIn('datatype_table', response_data)
    
    def test_analysis(self):
        with app.test_client() as client:
            with open('test.csv', 'rb') as f:
                data = {'file': (f, 'test.csv')}
                response_upload = client.post('/upload', data=data, content_type='multipart/form-data')
                response_data = json.loads(response_upload.data.decode('utf-8'))['data']
                data_dict = json.loads(response_data)
                column_x = list(data_dict.keys())[0]
                column_y = list(data_dict.keys())[1]
                response_analysis = client.post('/analysis', data={'data': response_data, 'target_column': column_y, 'action': 'check_correlation'})
                response_data = json.loads(response_analysis.data.decode('utf-8'))
                self.assertIn('data', response_data)
                self.assertIn('cols', response_data)
                self.assertIn('dict', response_data)
    
if __name__ == '__main__':
    unittest.main()
