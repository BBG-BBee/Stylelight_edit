## Pretrained models paths
# [인코더 모델 경로]: e4e 인코더 모델 파일 경로입니다. (최적화 기반 역전환 사용 시 필요 없음)
e4e = './pretrained_models/e4e_ffhq_encode.pt'             ##########

# [StyleGAN2 모델 경로]: 학습된 StyleGAN2 생성기(Generator) 모델 파일(.pkl)의 경로입니다.
# 사용자가 다운로드한 가중치 파일을 여기에 지정해야 합니다.
# 예: './pretrained_models/network-snapshot-002000.pkl'
stylegan2_ada_ffhq = './pretrained_models/network-snapshot-002000.pkl'             #######
 

style_clip_pretrained_mappers = ''
# [기타 모델 경로]: 얼굴 인식 및 정렬에 사용되는 보조 모델들의 경로입니다. (조명 추정 시 필수 아님)
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
# [결과 저장 경로]: 조명 추정 또는 편집 결과가 저장될 디렉토리입니다.
# 기본값: './assets/checkpoints_without_light_mask_both_finetuned'
# 결과 파일(.exr, .png 등)이 이 폴더에 생성됩니다.
checkpoints_dir = './assets/checkpoints_without_light_mask_both_finetuned'           ###### rebuttal
embedding_base_dir = './embeddings'                   #########
styleclip_output_dir = './StyleCLIP_results'
experiments_output_dir = './output'

## Input info
### Input dir, where the images reside
input_data_path = ''
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'barcelona'                ###########
save_image_path = './assets/save_videos/v5_2_visual'
save_video_path='assets/save_videos/project5_2.mp4'

## Keywords
pti_results_keyword = 'PTI'          ########
e4e_results_keyword = 'e4e'          #########
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'
