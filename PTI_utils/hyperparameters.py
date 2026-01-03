## Architechture
lpips_type = 'alex'      # LPIPS 손실 함수에 사용할 네트워크 유형 ('alex' 또는 'vgg')
first_inv_type = 'w'     # 초기 역전환(Inversion) 방식: 'w'는 최적화(Optimization) 기반, 'w+'는 인코더(Encoder) 기반 (현재 'w'로 설정됨)
optim_type = 'adam'      # 옵티마이저 알고리즘 (Adam 사용)

## Locality regularization (지역성 규제)
latent_ball_num_of_samples = 1       # 지역성 규제를 위해 샘플링할 잠재 벡터(latent code)의 개수
locality_regularization_interval = 1     ##### 지역성 규제를 적용할 간격 (매 스텝마다 적용)
use_locality_regularization = False      # 지역성 규제 사용 여부 (현재 False)

regulizer_alpha = 30     # 지역성 규제의 강도 (Alpha 값)

## Loss (손실 함수 가중치)
# pt_l2_lambda = 1 # org
# pt_l2_lambda = 10
# pt_lpips_lambda = 1

## Steps (학습 단계 설정)
LPIPS_value_threshold = 0.06    ##### LPIPS 손실 값이 이 임계값 이하로 떨어지면 PTI 튜닝을 조기 종료
max_pti_steps = 350           ###### PTI(Pivotal Tuning Inversion) 생성기 미세 조정(Fine-tuning) 최대 반복 횟수
first_inv_steps = 450                    ###### 초기 잠재 벡터(w)를 찾기 위한 투영(Projection/Optimization) 단계의 반복 횟수
max_images_to_invert = 30       ###### 역전환할 최대 이미지 수 (배치 처리 시 제한)

## Optimization (최적화 설정)
pti_learning_rate = 3e-4     ###### PTI 과정에서 생성기(Generator) 튜닝을 위한 학습률
first_inv_lr = 5e-3          ###### 초기 잠재 벡터(w) 최적화를 위한 학습률
train_batch_size = 1         # 학습 배치 크기
use_last_w_pivots = False    ###### 이전에 계산된 w 피벗(pivot)을 재사용할지 여부

############### 아래 옵션 중 하나를 선택: 조명 추정(Estimation) 또는 조명 편집(Editing) #############################

###################조명 추정: LFov LDR-> Panorama HDR ###########################
edit = False                 # 편집 모드 끄기 (추정 모드)
pt_l2_lambda = 10            # PTI 과정의 L2 손실 가중치
pt_lpips_lambda = 1          # PTI 과정의 LPIPS 손실 가중치
regulizer_l2_lambda = 0.01 #org 0.1  # 규제(Regularization) L2 가중치
regulizer_lpips_lambda = 0.01 # org 0.1 # 규제 LPIPS 가중치

################### 조명 편집: Lighting removal and addition ###########################
# edit = True                # 편집 모드 켜기
# pt_l2_lambda = 1
# pt_lpips_lambda = 1
# regulizer_l2_lambda = 0.01 #org 0.1
# regulizer_lpips_lambda = 0.01 # org 0.1
# edit = True
# edit_steps = 20            # 편집 단계 반복 횟수
# percentile = 0.9           # 조명 마스킹을 위한 백분위수 (상위 밝기 영역 추출용)
