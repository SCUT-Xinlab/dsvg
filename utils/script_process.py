
import sys
import os 
import time
import nibabel as nib
import subprocess
import csv
def load_nii(file_path):

    img = nib.load(file_path)
    data = img.get_fdata()  # 获取三维数据
    return data,img
def save_nii(data, reference_img, output_path):

    new_img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(new_img, output_path)
    print(f"文件已保存为: {output_path}")
class Subject:
    def __init__(self, folder_path):
        """
        通过文件夹路径初始化Subject类
        :param folder_path: str, 文件夹路径
        """
        self.folder_path = folder_path
        self.file_list = self.get_file_list(folder_path)
    @property
    def get_name(self):
        """
        获取文件夹的最后一层名称作为name
        :param folder_path: str, 文件夹路径
        :return: str, 文件夹的最后一层名称
        """
        return os.path.basename(self.folder_path)
    @property
    def age(self):
        import re
        match = re.search(r'(\d+)[Ww]', os.path.basename(self.folder_path))
        if match:
            return int(match.group(1))  # 提取数字部分并转换为整数
        else:
            match = re.search(r'(\d+)[Ww]', self.folder_path)
            if match:
                return int(match.group(1))
            else:
                return 0
    def get_result(self, output_folder):
        """
        根据output_folder和name生成一个包含文件路径的字典作为result
        :param output_folder: str, 输出文件夹路径
        :param name: str, subject的名称
        :return: dict, 包含不同路径的字典
        """
        name=self.get_name
        # 基础result文件夹路径
        result_folder = os.path.join(output_folder, name)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        # 生成结果路径的字典
        result_paths = {
            'volume': os.path.join(result_folder, f"{name}.nii.gz"),
            'model': os.path.join(result_folder, 'model'),
            'slice': os.path.join(result_folder, 'slice'),
            'sim-slice': os.path.join(result_folder, 'sim-slice')
        }

        return result_paths
    def get_volume_nodirectory(self, output_folder):
        """
        根据output_folder和name生成一个包含文件路径的字典作为result
        :param output_folder: str, 输出文件夹路径
        :param name: str, subject的名称
        :return: dict, 包含不同路径的字典
        """
        name=self.get_name
        # 基础result文件夹路径
        volume_file = os.path.join(output_folder,f"{name}.nii.gz")


        return volume_file 
    def get_mask_result(self, output_folder):
        name=self.get_name
        # 基础result文件夹路径
        result_folder = os.path.join(output_folder, name)
        mask_output_path=[]
        for filepath in self.file_list:  
            filename='mask_'+os.path.basename(filepath)
            mask_output_path.append(os.path.join(result_folder,filename))
        return result_folder,mask_output_path
    def get_mask_input(self,output_folder):
        if '-' in os.path.basename(self.folder_path):
            fname=os.path.basename(self.folder_path).split('-')[1]
        else:
            fname=os.path.basename(self.folder_path)
        mask_output_path=[]
        result_folder=os.path.join(output_folder,fname)
        for filepath in self.file_list:  
            filename=os.path.basename(filepath).replace('.nii.gz','')
            #maskfolder=os.path.join(result_folder,filename)
            #maskpath=os.path.join(maskfolder,filename+'_seg.nii.gz')
            maskpath=os.path.join(result_folder,filename+'_seg.nii.gz')
            mask_output_path.append( maskpath)
            #filename=os.path.basename(filepath).replace('.nii.gz','_seg.nii.gz')
            #mask_output_path.append(os.path.join(result_folder,filename))
        return result_folder,mask_output_path
    def is_processed(self, output_folder):
        """
        检查subject是否已经处理过
        :param output_folder: str, 输出文件夹路径
        :return: bool
        """
        name=self.get_name
        result_folder = os.path.join(output_folder, name)
        resample_volume=os.path.join(result_folder,"resample-vo.nii.gz")
        if os.path.exists(resample_volume) and os.path.getsize(resample_volume) > 0:
            return True
        result_volume=os.path.join(result_folder, f"{name}.nii.gz")
        if os.path.exists(result_volume) and os.path.getsize(result_volume) > 0:
            return True
        
        additonal_path=os.path.join(output_folder,f"{name}.nii.gz")
        if os.path.exists(additonal_path):
            return True
        return False
    def get_file_list(self, folder_path):
        """
        获取该文件夹下所有的 .nii 或 .nii.gz 文件路径
        :param folder_path: str, 文件夹路径
        :return: list, 该文件夹下的所有文件路径
        """
        files = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                files.append(file_path)
        return files
class DataSet:
    def __init__(self, data_file_path, output_folder, mode='process',mask_folder=None):
        """
        初始化方法，用于设置数据文件路径、输出文件夹和处理模式
        
        :param data_file_path: str, 数据文件路径
        :param output_folder: str, 输出文件夹路径
        :param mode: str, 处理模式（默认为'process'，可以根据需要更改）
        """
        self.data_file_path = data_file_path
        self.output_folder = output_folder
        self.mode = mode
        
        # 检查数据文件是否存在
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"数据文件未找到: {self.data_file_path}")
        
        # 如果输出文件夹不存在，则创建
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.visited_folders = set() 
        self.dataset= sorted(self.read_data_files(self.data_file_path),key=lambda x: x.get_name)
        self._resume=False
        self.maskfolder=mask_folder
    @property
    def resume(self):
        return self._resume
    @resume.setter
    def resume(self,value):
        assert value is not bool,"resume must be bool"
        self._resume=value
    @property
    def FBS_Seg(self):
        return self._FBS_Seg
    @FBS_Seg.setter
    def FBS_Seg(self,value):
        assert value is not bool,"resume must be bool"
        self._FBS_Seg=value
    def read_data_files(self, folder_path):
        """
        遍历数据文件夹，收集所有 Subject 对象及其文件路径。
        该函数支持以下几种文件夹结构：
        1. 所有文件存放在同一文件夹
        2. 文件按类别（如年龄、性别等）分类存放在子文件夹中
        3. 多层嵌套文件夹结构
        """
        subjects = []

        for root, dirs, files_in_dir in os.walk(folder_path):
            # 如果该文件夹下没有子文件夹，直接收集文件
            if root in self.visited_folders:
                continue
            if not dirs:

                subject = Subject(root)  # 创建 Subject 实例
                if subject.file_list:  # 只有当文件列表非空时，才加入
                    subjects.append(subject)
                self.visited_folders.add(root)

            else:
                # 遍历子文件夹并递归调用
                for subdir in dirs:
                    subdir_path = os.path.join(root, subdir)
                    subjects.extend(self.read_data_files(subdir_path))
                
                # 如果当前目录下有文件，收集它们
                subject = Subject(root)  # 创建 Subject 实例
                if subject.file_list:  # 只有当文件列表非空时，才加入
                    subjects.append(subject)

        return subjects
    def __len__(self):
        """
        返回数据集的长度
        :return: int, 数据集的元素数量
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        根据索引返回数据集中的元素
        :param index: int, 数据集元素的索引
        :return: object, 索引位置的元素
        """
        return self.dataset[index]
    def _nesvor_processing(self, item):
        """
        'nesvor'模式下的数据处理
        生成nesvor模式的命令行参数
        :param item: 当前的subject对象
        :return: list, 包含命令行参数的列表
        """
        # 创建input-stacks部分：遍历subject的file_list，添加到命令行列表
        input_stacks = ['--input-stacks']
        for file_path in item.file_list:
            input_stacks.append(file_path)

        # 创建output-volume部分：添加subject的result路径
        output_volume = ['--output-volume']
        output_volume.append(item.get_result(self.output_folder)['volume'])  # 假设'output-volume'是model路径

        # 合并命令行参数
        command = input_stacks + output_volume
        command+=['--device','0','--segmentation',"--output-resolution","0.5","--n-iter","6000","--batch-size","4096","--weight-image","1"]

        # 返回生成的命令行参数列表
        if not self.FBS_Seg:
            command.append("--nz_mask")
        return command
    def _svr_processing(self, item):
        """
        'nesvor'模式下的数据处理
        生成nesvor模式的命令行参数
        :param item: 当前的subject对象
        :return: list, 包含命令行参数的列表
        """
        # 创建input-stacks部分：遍历subject的file_list，添加到命令行列表
        input_stacks = ['--input-stacks']
        for file_path in item.file_list:
            input_stacks.append(file_path)

        # 创建output-volume部分：添加subject的result路径
        output_volume = ['--output-volume']
        output_volume.append(item.get_result(self.output_folder)['volume'])  # 假设'output-volume'是model路径

        # 合并命令行参数
        command = input_stacks + output_volume
        command+=['--device','3','--segmentation',"--output-resolution","0.5","--n-iter","3"]
        if not self.FBS_Seg:
            command.append("--nz_mask")
        return command
    def _fide_processing(self, item,device=0):
        """
        'nesvor'模式下的数据处理
        生成nesvor模式的命令行参数
        :param item: 当前的subject对象
        :return: list, 包含命令行参数的列表
        """
        # 创建input-stacks部分：遍历subject的file_list，添加到命令行列表
        input_stacks = ['--input-stacks']
        for file_path in item.file_list:
            input_stacks.append(file_path)
        input_masks=['--stack-masks']
        for mask_path in item.get_mask_input(self.maskfolder)[1]:
            input_masks.append(mask_path)
        # 创建output-volume部分：添加subject的result路径
        output_volume = ['--output-volume']
        output_volume.append(item.get_result(self.output_folder)['volume'])  # 假设'output-volume'是model路径

        # 合并命令行参数
        command = input_stacks +input_masks+ output_volume
        command+=['--device',f'{device}',"--output-resolution","0.5","--n-iter","3000","--batch-size","4096","--weight-image","1"]
        # 返回生成的命令行参数列表

        return command
    def _fide_nomask_processing(self, item):
        """
        'nesvor'模式下的数据处理
        生成nesvor模式的命令行参数
        :param item: 当前的subject对象
        :return: list, 包含命令行参数的列表
        """
        # 创建input-stacks部分：遍历subject的file_list，添加到命令行列表
        input_stacks = ['--input-stacks']
        for file_path in item.file_list:
            input_stacks.append(file_path)
        # 创建output-volume部分：添加subject的result路径
        output_volume = ['--output-volume']
        output_volume.append(item.get_result(self.output_folder)['volume'])  # 假设'output-volume'是model路径

        # 合并命令行参数
        command = input_stacks + output_volume
        command+=['--segmentation','--device','2',"--output-resolution","0.5","--n-iter","2000","--batch-size","4096","--weight-image","1"]
        if not self.FBS_Seg:
            command.append("--nz_mask")
        # 返回生成的命令行参数列表
        return command
    def _seg_processing(self, item):
        """
        'SEG'模式下的数据处理
        生成seg模式的命令行参数
        :param item: 当前的subject对象
        :return: list, 包含命令行参数的列表
        """
        assert self.maskfolder is not None,'need a path for stored mask'
        # 创建input-stacks部分：遍历subject的file_list，添加到命令行列表
        input_stacks = ['--input-stacks']
        for file_path in item.file_list:
            input_stacks.append(file_path)

        # 创建output-volume部分：添加subject的result路径
        output_volume = ['--output-stack-masks']
        output_volume.append(item.get_mask_result(self.maskfolder)[0])  # 假设'output-volume'是model路径

        # 合并命令行参数
        command = input_stacks + output_volume
        command+=['--device','0',"--batch-size-seg","16"]
        # 返回生成的命令行参数列表
        return command
    def _niftymic_processing(self, item):
        input_stacks = ['--filenames']
        for file_path in item.file_list:
            input_stacks.append(file_path)
        input_masks=[]
        if self.maskfolder:
            input_masks=['--filenames-masks']
            for mask_path in item.get_mask_result(self.maskfolder)[1]:
                input_masks.append(mask_path)

        output_volume = ['--output']
        output_volume.append(item.get_result(self.output_folder)['volume'])  # 假设'output-volume'是model路径

        command =['niftymic_reconstruct_volume']+ input_stacks +input_masks+ output_volume
        command+=['--outlier-rejection','1',"--isotropic-resolution","0.5","--subfolder-motion-correction","motion_correction","--intensity-correction","1"]

        return command
    def _svrtk_processing(self, item):
        input_stacks=[]
        for file_path in item.file_list:
            input_stacks.append(file_path)
        input_masks=[]
        if self.maskfolder:
            input_masks=['-mask',item.get_mask_result(self.maskfolder)[1][-1],'-template',input_stacks[-1]]
        command =['mirtk','reconstruct']+[item.get_result(self.output_folder)['volume']]+[str(len(input_stacks))]+input_stacks+input_masks
        command+=['-resolution', '0.5','-svr_only']
        return command
    def run(self, index):
        """
        根据self.mode的值处理dataset中的元素
        :param index: int, 数据集元素的索引
        :return: processed_item, 处理后的数据项
        """
        item = self[index]  # 获取对应的元素
        if self.resume and item.is_processed(self.output_folder):
            return None
        
        if self.mode == 'help':
            print('To run a new command ,just add a new processing function')
            pass
        elif self.mode == 'nesvor':
            # 处理nesvor模式：生成命令行参数
            processed_item = self._nesvor_processing(item)  
        elif self.mode == 'fide':
            # 处理nesvor模式：生成命令行参数
            processed_item = self._fide_processing(item)
        elif self.mode == 'seg':
            # 处理nesvor模式：生成命令行参数
            processed_item = self._seg_processing(item)
        elif self.mode=='fide_nonemask':
            processed_item=self._fide_nomask_processing(item)
        elif self.mode=='niftymic':
            processed_item=self._niftymic_processing(item)   
        elif self.mode=='svr':
            processed_item=self._svr_processing(item)
        elif self.mode=='svrtk':
            processed_item=self._svrtk_processing(item)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return processed_item
    def age(self,index):
        item = self[index]
        return item.age
    def debug(self,name):
        print(name)
        for item in self.dataset:
            if item.get_name()==name:
                processed_item = self._nesvor_processing(item)
                return processed_item
            else:
                continue
        print(f"no such subject named {name}")
        return None
    def seg_greater_zero(self, index):
        item = self[index]
        for file_path in item.file_list:
            data,img=load_nii(file_path=file_path)
            data=(data>0).astype(float)
            outfolder=os.path.join(self.output_folder,item.get_name)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            filename='mask_'+os.path.basename(file_path)
            save_nii(data,img,os.path.join(outfolder,os.path.join(outfolder,filename)))
def datesetloader(input_folder,output_folder,maskfolder=None,mode='train',resume=False):
    dataset=DataSet(input_folder,output_folder,mode=mode,mask_folder=maskfolder)
    dataset.resume=resume
    return dataset
def save_execution_time_to_csv(name,start_time, end_time, filename="execution_time.csv"):
    """
    保存执行时间到 CSV 文件
    """
    execution_time = end_time - start_time

    # 打开 CSV 文件，追加时间
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name,execution_time])  # 每次写入一行时间值

    return execution_time
def main(dataset:DataSet):
    from nesvor.cli.parsers import main_parser
    from nesvor.cli.main import run
    for i in range(len(dataset)):
        command=dataset.run(i)
        if command is None:
            print("no need for reconstructed subject in resume process")
            continue
        sys.argv.clear()
        #TODO: different command
        sys.argv.extend(['nesvor','reconstruct'])
        sys.argv.extend(command)
        parser, subparsers = main_parser()
        args = parser.parse_args()
        try:
            run(args)
        except Exception as e:
            print(e)
            print("error in subject:",dataset[i].get_name)
            continue
        
def main_SVR(dataset:DataSet):
    from nesvor.cli.parsers import main_parser
    from nesvor.cli.main import run
    for i in range(len(dataset)):
        command=dataset.run(i)
        if command is None:
            print("no need for reconstructed subject in resume process")
            continue
        sys.argv.clear()
        #TODO: different command
        sys.argv.extend(['nesvor','svr'])
        sys.argv.extend(command)
        parser, subparsers = main_parser()
        args = parser.parse_args()
        try:
            run(args)
        except Exception as e:
            print(e)
            print("error in subject:",dataset[i].get_name)
            continue
def main_niftymic(dataset:DataSet):
    csv_file=os.path.join(dataset.output_folder,'time.csv')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["文件",  "time"])
    for i in range(len(dataset)):

        command=dataset.run(i)
        if command is None:
            print("no need for reconstructed subject in resume process")
            continue
        start_time = time.time()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        end_time = time.time()

        stdout = result.stdout
        stderr = result.stderr
        with open(os.path.dirname(dataset[i].get_result(dataset.output_folder)['volume'])+'/system_out.txt', 'a') as file:

            file.write(stdout)
            file.write(stderr)
        execution_time = save_execution_time_to_csv(dataset[i].get_name,start_time, end_time,filename=csv_file)
def sim_dataset_get( mode='fide_nonemask',sim_out=None):
    sim_in='/home/lvyao/local/sim/LR_random_large'
    if sim_out is None:
        sim_out='/home/lvyao/local/sim/result/'+mode+'_noatlas_dr3'
    resume=False
    maskpath='/home/lvyao/local/sim/seg'
    return datesetloader(sim_in,sim_out,mode=mode,resume=resume,maskfolder=maskpath)
def lv_dataset_get(mode='niftymic',path_in='/home/lvyao/local/lv(115)',path_out='/home/lvyao/local/fideresult/lv_fide-noaltas'):
    mask_folder='/home/lvyao/local/lv_seg'
    #mode='svrtk'
    mode=mode
    resume=True
    return datesetloader(path_in,path_out,mode=mode,resume=resume,maskfolder=mask_folder)

def guo_dataset_get():
    #path_in='/home/lvyao/local/guo/2025.1.3'
    #path_in='/home/lvyao/local/guo/guo-1'
    path_in='/home/lvyao/local/guo/guo-1'
    path_out='/home/lvyao/local/guo/resultsvr'
    mask_folder='/home/lvyao/local/sim/seg'
    mode='nesvor'
    resume=True
    return datesetloader(path_in,path_out,mode=mode,resume=resume,maskfolder=mask_folder)

def guo_dataset_get():
    #path_in='/home/lvyao/local/guo/2025.1.3'
    #path_in='/home/lvyao/local/guo/guo-1'
    path_in='/home/lvyao/local/guo/guo-1'
    path_out='/home/lvyao/local/guo/resultsvr'
    mask_folder='/home/lvyao/local/sim/seg'
    mode='svr'
    resume=True
    return datesetloader(path_in,path_out,mode=mode,resume=resume,maskfolder=mask_folder)
def simseg_dataset():
    path_in='/home/lvyao/local/sim/LR_random_large'
    path_out='/home/lvyao/local/sim/seg'
    mode='niftymic'
    resume=True
    dataset=datesetloader(path_in,path_out,mode=mode,resume=resume)
    for i in range(len(dataset)):
        dataset.seg_greater_zero(i)
def yang_dataset_get():
    yang_in='/home/lvyao/nesvor/dataset/nii_stack/yang_final（20-38W）464'
    yang_out='/home/lvyao/nesvor/output-all/yang-nesvor'
    mode='nesvor'
    resume=True
    return datesetloader(yang_in,yang_out,mode=mode,resume=resume)
def post_process():
    import torch
    from nesvor.image import Volume, Slice, load_volume
    from nesvor.transform import RigidTransform
    root_folder = '/home/lvyao/local/fideresult/lv'
    output_folder = '/home/lvyao/local/fideresult/lv_post'
    # 目标 transformation 矩阵
    new_transformation=RigidTransform(data=torch.tensor(
    [[[1.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00],
    [0.0000e+00, 1.0000e+00,  0.0000e+00, 0.0000e+00],
    [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00]]],device='cpu'))
    # atlas= load_volume('/home/lvyao/Git_157copy/Git/S-R-data/CRL_FetalBrainAtlas_2017v3/STA36exp.nii.gz')
    # new_transformation=atlas.transformation
    resample_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == 'resample-vvo.nii.gz':
                resample_files.append(os.path.join(root, file))

    # 处理每个文件
    for file_path in resample_files:
        folder_name = os.path.basename(os.path.dirname(file_path))
    
        # 生成输出文件路径
        output_file_path = os.path.join(output_folder, f"{folder_name}.nii.gz")
        print(f"Processing file: {file_path}")
        
        # 加载 volume
        volume = load_volume(file_path)
        
        # 修改 transformation 属性
        volume.transformation = new_transformation

        volume.save(output_file_path)
        print(f"saved file: {file_path}")
def feta_dataset_get( mode='fide_nonemask',sim_out=None):
    sim_in='/home/lvyao/local/sim/feta'
    if sim_out is None:
        sim_out='/home/lvyao/local/sim/feta_result/'+mode+'_noatlas_dr3'
    resume=False
    return datesetloader(sim_in,sim_out,mode=mode,resume=resume)

def xu_dataset_get():
    path_in='/home/lvyao/local/xu/dataset'
    path_out='/home/lvyao/local/xu/result'
    mode='nesvor'
    resume=False
    return datesetloader(path_in,path_out,mode=mode,resume=resume)
if __name__ == "__main__":
    dataset=sim_dataset_get()
    dataset.FBS_Seg=True
    main(dataset)
