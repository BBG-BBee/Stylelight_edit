from random import choice
from string import ascii_uppercase
# from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os

# CUDA 환경 설정 (매번 자동 적용)
from setup_env import setup_cuda_environment
setup_cuda_environment(verbose=True)  # False는 환경 설정 메시지 최소화

from PTI_utils import global_config, paths_config, hyperparameters
import wandb
from training.coaches.my_coach import MyCoach
from training.coaches.my_editor import MyEditor
from PTI_utils.ImagesDataset import ImagesDataset

import glob




def run_PTI(run_name='', use_wandb=False):
    """
    PTI (Pivotal Tuning Inversion) 프로세스를 실행하는 메인 함수입니다.
    설정에 따라 조명 추정(MyCoach) 또는 조명 편집(MyEditor)을 수행합니다.
    """
    # CUDA 디바이스 설정: GPU 사용 순서와 사용할 GPU 번호를 환경 변수에 설정
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    # 실행 이름(run_name)이 지정되지 않은 경우, 랜덤한 12자리 영문 대문자 문자열 생성
    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    # Weights & Biases (wandb) 사용 여부에 따른 초기화
    # 실험 기록 및 시각화를 위해 사용됩니다.
    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    
    # PTI 학습 관련 전역 설정 초기화
    # pivotal_training_steps: 튜닝 스텝 수 (여기서는 1로 초기화되지만 내부 루프에서 변경될 수 있음)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    # embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    # os.makedirs(embedding_dir_path, exist_ok=True)
    
    # 결과 이미지를 저장할 디렉토리 생성
    os.makedirs(paths_config.save_image_path, exist_ok=True)


    # dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # hyperparameters.edit 값에 따라 모드 결정 (조명 추정 vs 조명 편집)
    if not hyperparameters.edit:
        # [조명 추정 모드 (Lighting Estimation)]
        # IndoorHDRDataset 등의 테스트 데이터셋 경로 설정
        # root_path = '/home/deep/projects/mini-stylegan2/Evaluation/data/ground_truth_ours_neg0.6_60degree_HR/crop_test_high_resolution/*png'
        root_path = './before_image/'
        
        # 데이터 로더 설정: 파일 경로 리스트를 가져와서 사용 - 경로 내 모든 파일을 처리하도록 수정함
        # jpg와 png 파일 모두 가져오기 (대소문자 구분 없이 처리하기 위해 여러 패턴 시도하거나 간단히 주요 확장자만 지정)
        dataloader = sorted(glob.glob(root_path + '*jpg') + glob.glob(root_path + '*png') + glob.glob(root_path + '*jpeg'))
        # dataloader = sorted(glob.glob(root_path)) 

        # root_path = 'assets/wild2/*jp*g'
        # dataloader = sorted(glob.glob(root_path))

        # MyCoach: 조명 추정 및 파노라마 생성을 담당하는 코치 클래스 인스턴스화
        coach = MyCoach(dataloader, use_wandb)

    else:
        # [조명 편집 모드 (Lighting Editing)]
        # 편집할 테스트 이미지 데이터셋 경로 설정
        # root_path = 'assets/test_set_light_editing_new/*195*png'
        # root_path = 'assets/test_set_light_editing_new/*205*png'
        root_path = 'assets/test_set_light_editing_new/*279*png'
        
        # 데이터 로더 설정: 파일 경로 리스트 가져오기
        dataloader = sorted(glob.glob(root_path))#[:10]
        
        # MyEditor: 조명 편집을 담당하는 에디터 클래스 인스턴스화
        coach = MyEditor(dataloader, use_wandb)


    # 설정된 코치(MyCoach 또는 MyEditor)를 사용하여 학습/추론 시작
    coach.train()

    return global_config.run_name


if __name__ == '__main__':  
    run_PTI(run_name='', use_wandb=False)