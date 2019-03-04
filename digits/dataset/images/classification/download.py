# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os, sys, socket, re
import ast, json, time, signal
import requests
import zipfile
import logging

from ..job import ImageDatasetJob
from digits.task import Task
from digits.dataset import tasks
from digits.status import Status

from digits import utils
from digits.utils import subclass, override, constants


PICKLE_VERSION = 2


@subclass
class DownToFolderTask(Task):

    def __init__(self, job_name, project_info, folder, **kwargs):
        self.job_name = job_name
        self.project_info = project_info
        self.folder = folder
        self.progress = float(0)
        

        super(DownToFolderTask, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(DownToFolderTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(DownToFolderTask, self).__setstate__(state)

    @override
    def name(self):
        return 'download/unzip from http'

    @override
    def html_id(self):
        return 'task-down-folder'

    @override
    def offer_resources(self, resources):
        key = 'parse_folder_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources, env):
        args = [ "nvidia-smi" ]
        return args

    # customized code
    @override
    def run(self, resources):
        if self.status == Status.DONE:
            return True

        self.before_run()
        self.logger.info('%s task started.' % self.name())
        self.status = Status.RUN

        unrecognized_output = []

        project_info = self.project_info

        headers = {'Content-Type': 'application/json; charset=utf-8',
                'Authorization': 'Token 32c25086cbaf8bc9498ebb4609ec9cb602613473', }
        url = 'http://192.168.0.180:8080/project/' + str(project_info['pk']) + "/download/"
        body = str(project_info['id_list'])
        res = requests.post(url, headers=headers, data=body, stream=True)

        if not res.status_code == 200:
            raise ValueError("not exist project", res.status_code)

        zip_path = os.path.join("/mnt/data", 
                                # project_info['title'] + "_to_" +self.job_name, 
                                self.job_name,
                                self.job_name + ".tar")

        if not os.path.exists(os.path.dirname(zip_path)):
            os.makedirs(os.path.dirname(zip_path))
        
        f = open(zip_path, 'wb')
        f.write(res.content)
        f.close()
        
        zip_file = zipfile.ZipFile(zip_path)
        zip_file.extractall(os.path.dirname(zip_path))
        zip_file.close()
        os.remove(zip_path)

        self.return_code = 0
        self.after_run()

        self.logger.info('%s task completed.' % self.name())
        self.status = Status.DONE
        return True

def receive_strings_from(conn):
    while True:
        chunk = conn.recv(4000)
        if len(chunk) == 0:
            break
        yield chunk.decode('utf-8')

@subclass
class CreateListFileTask(Task):
    
    def __init__(self, job_name, total_count, **kwargs):
        self.job_name = job_name
        self.src_dir = os.path.join('/','mnt','data', job_name)
        self.dst_dir = os.path.join('/','mnt','jobs', job_name)        
        self.progress = float(0)
        self.total_count = total_count

        super(CreateListFileTask, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(CreateListFileTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(CreateListFileTask, self).__setstate__(state)

    @override
    def name(self):
        return 'create label/list file'

    @override
    def html_id(self):
        return 'task-create-list'

    @override
    def offer_resources(self, resources):
        key = 'parse_folder_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None


    # customized code
    @override
    def run(self, resources):
        if self.status == Status.DONE:
            return True

        self.before_run()
        self.logger.info('%s task started.' % self.name())
        self.status = Status.RUN

        unrecognized_output = []

        import sys, httplib, json
        import StringIO

        logging.warning(self.name())

        request_data = [
            "create_list_file",
            self.src_dir,
            self.dst_dir,
            self.total_count,
            self.job_id
        ]

        host_uri = 'remote_caffe'
        host_port = 17219

        s = socket.create_connection((host_uri, host_port))
        assert s != -1
        
        j = json.dumps(request_data)

        # b = bytes(j, encoding='utf-8') py3
        b = bytes(j)
        assert len(b) > 0
        s.send(b)
        s.shutdown(socket.SHUT_WR)
        
        import StringIO

        buf = receive_strings_from(s)
        try:
            sigterm_time = None  # When was the SIGTERM signal sent
            sigterm_timeout = 2  # When should the SIGKILL signal be sent
            for line in buf:
                # self.process_output(line)
                if self.aborted.is_set():
                    if sigterm_time is None:
                        # Attempt graceful shutdown
                        self.p.send_signal(signal.SIGTERM)
                        sigterm_time = time.time()
                        self.status = Status.ABORT
                    break
                if line is not None:
                    # Remove whitespace
                    line = line.strip()

                if line:
                    if 'progress' in line:
                        line = line.strip("\n")
                        self.process_output(line)
                        # self.logger.warning('%s unrecognized output: %s' % (self.name(), line.strip()))
                        unrecognized_output.append(line)
                else:
                    time.sleep(0.05)

            if sigterm_time is not None and (time.time() - sigterm_time > sigterm_timeout):
                self.p.send_signal(signal.SIGKILL)
                self.logger.warning('Sent SIGKILL to task "%s"' % self.name())
                time.sleep(0.1)
            time.sleep(0.01)
        except:
            self.p.terminate()
            self.after_run()
            raise

        self.after_run()

        s.close()        
        self.return_code = 0
        self.logger.info('%s task completed.' % self.name())
        self.status = Status.DONE
        return True


    @override
    def task_arguments(self, resources, env):
        args = [ "nvidia-smi" ]
        return args

    @override
    def process_output(self, line):
        self.progress = float(line.split('progress/')[-1]) / self.total_count
        self.emit_progress_update()
        return True


@subclass
class CreateCaffeDbTask(Task):
    
    def __init__(self, job_name, **kwargs):
        self.job_name = job_name
        self.progress = float(0)
        self.current_count = 0
        self.pre_count = 0
        self.total_count = kwargs.pop('total_count', None)
        self.encoding = kwargs.pop('encoding', None)
        self.compression = kwargs.pop('compression', None)
        self.backend = kwargs.pop('backend', None)
        super(CreateCaffeDbTask, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(CreateCaffeDbTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(CreateCaffeDbTask, self).__setstate__(state)

    @override
    def name(self):
        return 'create_caffe_lmdb'

    @override
    def html_id(self):
        return 'task-create-db'

    @override
    def offer_resources(self, resources):
        key = 'parse_folder_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def run(self, resources):
        self.before_run()
        self.status = Status.RUN

        data_path = os.path.join(os.path.join("/", "mnt", "data"),
                                self.job_name)
        workspace = os.path.join(os.path.join("/", "mnt", "jobs"),
                                self.job_name)

        split_names = ["train", "val", "test"]
        split_ratio = [100 - 26 - 23, 26, 23]

        logging.warning(self.name())

        request_data = [
            "create_lmdb",
            self.job_name,
            data_path,
            workspace,
            256,
            256,
            split_names,
            split_ratio
        ]

        host_uri = 'remote_caffe'
        host_port = 17219

        s = socket.create_connection((host_uri, host_port))
        assert s != -1

        # b = bytes(json.dumps(request_data), encoding='utf-8') # py3
        b = bytes(json.dumps(request_data))
        assert len(b) > 0
        s.send(b)
        s.shutdown(socket.SHUT_WR)

        for m in receive_strings_from(s):
            self.process_output(m)
        s.close()        

        self.after_run()
        self.status = Status.DONE
        return True

    @override
    def task_arguments(self, resources, env):
        args = [ "nvidia-smi" ]
        return args

    @override
    def process_output(self, line):

        if 'convert_imageset' in line and 'Processed' in line: 
            match = re.match(r'(\d+) files.', line.split(' Processed ')[1])
            if match:
                cnt = int(match.group(1))
                
                if self.pre_count < cnt:
                    self.current_count += cnt-self.pre_count
                    self.pre_count = cnt
                else:
                    self.pre_count = cnt
                    self.current_count += cnt
                self.progress = float(self.current_count)/self.total_count
        
        self.emit_progress_update()

        return True




@subclass
class DownloadDatasetJob(ImageDatasetJob):
    """
    A Job that creates an image dataset for a classification network
    """

    def __init__(self, **kwargs):
        self.download_uri = kwargs.pop('uri', None)
        form = kwargs.pop('form', None)

        self.labels_file = None
        super(DownloadDatasetJob, self).__init__(**kwargs)
        
        
        self.pickver_job_dataset_image_classification = PICKLE_VERSION
        logging.info("{} __init__".format(self.id()))

    def __setstate__(self, state):
        super(DownloadDatasetJob, self).__setstate__(state)

        logging.info("{} __setstate__".format(self.id()))

        self.pickver_job_dataset_image_classification = PICKLE_VERSION

    def create_db_tasks(self):
        """
        Return all CreateDbTasks for this job
        """
        # return [t for t in self.tasks if isinstance(t, tasks.CreateDbTask)]
        return []

    @override
    def get_backend(self):
        """
        Return the DB backend used to create this dataset
        """
        return self.caffe_db_task().backend

    @override
    def get_entry_count(self, stage):
        """
        Return the number of entries in the DB matching the specified stage
        """
        if stage == constants.TRAIN_DB or stage == 'train':
            db = self.get_image_count('train')
        elif stage == constants.VAL_DB or stage == 'val':
            db = self.get_image_count('val')
        elif stage == constants.TEST_DB or stage == 'test':
            db = self.get_image_count('test')
        else:
            return 0
        # return db.entries_count if db is not None else 0
        return db 

    def get_image_count(self, type):
        try:
            file_path = os.path.join(self.dir(), type+'.txt')
            lines = open(file_path, 'r').read().split('\n')
            lines.remove('')
            cnt = len(lines)
        except:
            cnt = 0
        return cnt
        
    @override
    def get_feature_dims(self):
        """
        Return the shape of the feature N-D array
        """
        return self.image_dims

    def get_encoding(self):
        """
        Return the DB encoding used to create this dataset
        """
        return self.caffe_db_task().encoding

    def get_compression(self):
        """
        Return the DB compression used to create this dataset
        """
        return self.caffe_db_task().compression

    @override
    def get_feature_db_path(self, stage):
        """
        Return the absolute feature DB path for the specified stage
        """
        path = self.path(stage)
        return path if os.path.exists(path) else None

    @override
    def get_label_db_path(self, stage):
        """
        Return the absolute label DB path for the specified stage
        """
        # classification datasets don't have label DBs
        return None

    @override
    def get_mean_file(self):
        """
        Return the mean file
        """
        return os.path.join(self.dir(), "mean.binaryproto")

    @override
    def job_type(self):
        return 'Alturk Image Classification Dataset'

    @override
    def json_dict(self, verbose=False):
        d = super(ImageClassificationDatasetJob, self).json_dict(verbose)
        if verbose:
            d.update({
                'ParseFolderTasks': [{
                    "name":        t.name(),
                    "label_count": t.label_count,
                    "train_count": t.train_count,
                    "val_count":   t.val_count,
                    "test_count":  t.test_count,
                } for t in self.parse_folder_tasks()],
                'CreateDbTasks': [{
                    "name":             t.name(),
                    "entries":          t.entries_count,
                    "image_width":      t.image_dims[0],
                    "image_height":     t.image_dims[1],
                    "image_channels":   t.image_dims[2],
                    "backend":          t.backend,
                    "encoding":         t.encoding,
                    "compression":      t.compression,
                } for t in self.create_db_tasks()],
            })
        return d

    def parse_folder_tasks(self):
        """
        Return all ParseFolderTasks for this job
        """
        # return [t for t in self.tasks if isinstance(t, tasks.ParseFolderTask)]
        return []

    def list_file_task(self):
        """
        Return the task that creates the test set
        """
        for t in self.tasks:
            if isinstance(t, CreateListFileTask):
                return t

    def caffe_db_task(self):
        """
        Return the task that creates the training set
        """
        for t in self.tasks:
            if isinstance(t, CreateCaffeDbTask):
                return t
        

    # def val_db_task(self):
    #     """
    #     Return the task that creates the validation set
    #     """
    #     for t in self.tasks:
    #         if isinstance(t, tasks.CreateDbTask) and 'val' in t.name().lower():
    #             return t
    #     return None