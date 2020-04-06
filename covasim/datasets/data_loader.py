import requests
import os
import json
dirname = os.path.dirname(__file__)


class DataLoader:
    def __init__(self):
        pass

    def update_data(self):
        r = requests.get(self.URL)
        data_json = r.json()
        self.delete_data()
        translated = self.translate(data_json)
        with open(self.file_path(), "a") as outfile:
            json.dump(translated, outfile)
        self.data = data_json

    def delete_data(self):
        if os.path.exists(self.file_path()):
            os.remove(self.file_path())

    def file_path(self):
        return os.path.join(dirname,"sources/population/{0}.json".format(self.FILENAME))

    def load_data(self):
        if hasattr(self, 'data'):
            return
        with open(self.file_path()) as datafile:
            strdata = json.load(datafile)
        self.data = strdata

    def data_for_country(self, country):
        self.load_data()
        return self._for_country(country)

    def _for_country(self, country):
        raise NotImplementedError

    def translate(self, json):
        raise NotImplementedError
