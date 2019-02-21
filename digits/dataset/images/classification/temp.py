# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os, logging
import ast
import requests
import zipfile
import json

from .job import ImageClassificationDatasetJob
from digits.dataset import tasks
from digits.status import Status
from digits.utils import subclass, override, constants

logging.info(requests.__version__)

def alturk_project_download(project_info, form):
    logging.fatal("alturk_project_download")

    headers = {'Content-Type': 'application/json; charset=utf-8',
               'Authorization': 'Token 32c25086cbaf8bc9498ebb4609ec9cb602613473', }
    url = 'http://192.168.0.180:8080/project/' + str(project_info['pk']) + "/download/"
    body = str(project_info['id_list'])
    res = requests.post(url, headers=headers, data=body, stream=True)

    if not res.status_code == 200:
        raise ValueError("not exist project", res.status_code)

    zip_path = os.path.join("/app/data", 
                            project_info['title'] + "_to_" +form.dataset_name.data, 
                            form.dataset_name.data + ".tar")

    if not os.path.exists(os.path.dirname(zip_path)):
        os.makedirs(os.path.dirname(zip_path))
    
    f = open(zip_path, 'wb')
    f.write(res.content)
    f.close()

    zip_file = zipfile.ZipFile(zip_path)
    zip_file.extractall(os.path.dirname(zip_path))
    zip_file.close()
    os.remove(zip_path)
    return os.path.dirname(zip_path)


def alturk_project_formatting(data_path):
    logging.fatal("alturk_project_formatting")

    info = {}
    try:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                fname, ext = os.path.splitext(file)
                if ext == ".json": continue
                json_path = os.path.join(root, fname+".json")
                with open(json_path) as f:
                    loaded_data = json.loads(f.read())
                if 'classes' in loaded_data:
                    label = str(loaded_data['classes'][0])
                    move_folder = os.path.join(data_path, label)
                    move_path = os.path.join(move_folder, file)
                    if not os.path.exists(move_folder):
                        os.mkdir(move_folder)
                    os.rename(file_path, move_path)
                    os.remove(json_path)

                    if label not in info:
                        info[label] = []
                    info[label].append(move_path)

    except:
        return False

    label_txt = open(os.path.join(data_path, "labels.txt"), 'w')
    train_txt = open(os.path.join(data_path, "train.txt"), "w")
    label_index = 0
    for label in sorted(info):
        label_txt.write(label+ "\n")
        for line in info[label]:
            train_txt.write(line + " " + str(label_index) +"\n")
        label_index += 1

    label_txt.close()
    train_txt.close()

    return True


def from_alturk(job, form):
    project_info = ast.literal_eval(form.alturk_project_download.data)

    # alturk_project_download
    folder_path = alturk_project_download(project_info, form)
    if not alturk_project_formatting(folder_path):
        raise ValueError('formatting error')

    form.folder_train.data = folder_path
    from_folders(job,form)


@subclass
class DownloadDatasetJob(ImageClassificationDatasetJob):

    def __init__(self, uri, form, **kwargs):
        super(ImageClassificationDatasetJob, self).__init__(**kwargs)
    #     self.download_uri = uri
    #     self.downloads = []
    #     # self.downloads.append(tasks.DownloadZipTask(self.download_uri))
    #     self.entry_count = 12345
    #     # project_info = ast.literal_eval(form.alturk_project_download.data)
    #     # logging.fatal("alturk_project_download")
    #     # # alturk_project_download
    #     # folder_path = alturk_project_download(project_info, form)
    #     # if not alturk_project_formatting(folder_path):
    #     #     raise ValueError('formatting error')
    #     # form.folder_train.data = folder_path
    #     # from_folders(job,form)
        assert self._id != None
        self.job_id = self._id 
        assert self.job_id != None
        assert self.tasks != None
        ts = []
        ts.append(tasks.DownloadZipTask("www.google.com", self.job_id))
        for t in self.tasks:
            ts.append(t)

        self.tasks = ts


    def __setstate__(self, state):
        tasks.DownloadZipTask()
        assert False

        super(ImageClassificationDatasetJob, self).__setstate__(state)
        

    @override
    def job_type(self):
        return 'Download And Create Image Dataset'

    # @override
    # def json_dict(self, verbose=False):
    #     d = { 'AnalyzeDbTask': { "name": "undefined" } }
    #     return d
