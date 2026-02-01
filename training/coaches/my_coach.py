import os
import torch
from tqdm import tqdm
from PTI_utils import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from PTI_utils.log_utils import log_images_from_w

from skylibs.envmap import EnvironmentMap
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw
import imageio
import glob

# from skylibs.envmap import EnvironmentMap
from skylibs.demo_crop import crop2pano
from skylibs.hdrio import imread, imsave

# TTA 관련 import
from training.tta_augment import TTAAugmentor, ExpAug, WBAug, FlipAug, IdentityAug
from training.s2r_adapter import (
    compute_uncertainty_from_outputs,
    apply_adaptive_scales,
    set_adapter_scales,
    get_adapter_scales
)

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha,tonemapped_img



class MyCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        use_ball_holder = True
        is_128x256 = False

        # for fname, image in tqdm(self.data_loader):
        for image_name in tqdm(self.data_loader):
            env = EnvironmentMap(256, 'latlong')
            image_crop2pano = crop2pano(env, image_name)
            print('image max:', image_crop2pano.max())
            image = 2*(image_crop2pano/255.0)-1


            image = torch.tensor(image.transpose([2, 0, 1]), device=global_config.device)#/255.0     ################################### 0-1
            image = image.unsqueeze(0).to(torch.float32)


            self.restart_training()
            # name = image_name.split('/')[-1].split('.')[0]
            name = os.path.splitext(os.path.basename(image_name))[0]

            # mask_fname = '/home/deep/projects/mini-stylegan2/crop10.jpg'
            # mask_fname = '/home/deep/projects/mini-stylegan2/crop60.jpg'
            if is_128x256:
                mask_fname = '/home/deep/projects/mini-stylegan2/crop60.jpg'
            else:
                mask_fname = 'crop60_256x512.jpg'
            mask_pil = PIL.Image.open(mask_fname).convert('RGB')
            use_debug = False
            if use_debug:
                print('###mask_pil size:', np.array(mask_pil).shape)
            
            if is_128x256:
                mask_pil = mask_pil.resize((256, 128), PIL.Image.LANCZOS)
            else:
                mask_pil = mask_pil.resize((512, 256), PIL.Image.LANCZOS)


            mask_pil_sum_c=np.sum(mask_pil,axis=2)
            mask_pil_sum_c_row = np.sum(mask_pil_sum_c,axis=1)
            mask_pil_sum_c_col = np.sum(mask_pil_sum_c,axis=0)
            row_min = np.argwhere(mask_pil_sum_c_row).min()+10 #128
            row_max = np.argwhere(mask_pil_sum_c_row).max()-5

            col_min = np.argwhere(mask_pil_sum_c_col).min()+10 #256
            col_max = np.argwhere(mask_pil_sum_c_col).max()-10  
            
            img1 = ImageDraw.Draw(mask_pil) 
            img1.rectangle([(col_min,row_min),(col_max, row_max)],fill=(255, 0, 0), outline ="red") 

            # im1 = mask_pil.crop((col_min, row_min, col_max, row_max))
            if use_debug:
                mask_pil.save(f'debug_bbox_{name}.png')

            bbox = [row_min, row_max, col_min, col_max] 

            # [초점 마스킹 GAN 역전환 (Focus Masking GAN Inversion)]: 
            # 입력 LDR 이미지의 제한된 시야각(LFov)과 하이라이트 영역에 집중하여, 
            # 해당 이미지를 가장 잘 복원해내는 잠재 벡터 w를 찾는 최적화 과정입니다.
            w_pivot = None
            w_pivot = self.calc_inversions(image, name, bbox) ## here
            w_pivot = w_pivot.to(global_config.device)
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            real_images_batch = real_images_batch[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
            
            use_first_phase = False
            if use_first_phase:
                generated_images = self.forward(w_pivot)
                is_png = True
                if is_png:
                    generated_images = torch.clip(generated_images, -1, 1)
                    generated_images = (generated_images + 1) * (255/2)
                    generated_images = generated_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    PIL.Image.fromarray(generated_images, 'RGB').save(f'{paths_config.checkpoints_dir}/{name}.png')
                else:
                    gamma = 2.4
                    hdr = torch.clip(generated_images, -1, 1)
                    full = (hdr+1)/2
                    tone = True
                    if tone:
                        full_inv_tonemap = torch.pow(full/5, gamma)
                        img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy() 
                    else:
                        img_hdr_np = full.permute(0, 2, 3, 1)[0].cpu().numpy()


                    imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)


            do_save_image = False
            for i in tqdm(range(hyperparameters.max_pti_steps)):   #max_pti_steps = 350
                generated_images = self.forward(w_pivot)
                
                
                if do_save_image:
                    generated_images_ = (generated_images + 1) * (255/2)
                    generated_images_ = generated_images_.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    PIL.Image.fromarray(generated_images_, 'RGB').save(f'{paths_config.save_image_path}/{hyperparameters.first_inv_steps+i:04}.png')

                generated_images = torch.clip(generated_images, -1, 1)                     ######
                # [LFov 마스크 (LFov Mask)]: 생성된 이미지를 입력 이미지의 시야각(Bounding Box)에 맞춰 잘라내어 비교
                generated_images = generated_images[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]

                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:     #LPIPS_value_threshold = 0.06 
                    break

                loss.backward()
                self.optimizer.step()
                
                #locality_regularization_interval = 1     #####
                #training_step = 1
                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0   ##
                
                # image_rec_result_log_snapshot = 100
                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            generated_images = self.forward(w_pivot)
            is_png = False # in rebuttal, we use False
            if is_png:
                generated_images = (generated_images + 1) * (255/2)
                generated_images = generated_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(generated_images, 'RGB').save(f'{paths_config.checkpoints_dir}/{name}_test.png')
            else:
                gamma = 2.4
                limited = True

                if limited:
                    generated_images_singlemap = torch.mean(generated_images, dim=1, keepdim=True)                                                                             
                    r_percentile = torch.quantile(generated_images_singlemap,0.999)                                                                                     
                    light_mask = (generated_images_singlemap > r_percentile)*1.0                                                                                    
                    hdr = torch.clip(generated_images*(1-light_mask), -1, 1)+torch.clip(generated_images*light_mask, -1, 2)             

                else:
                    hdr = torch.clip(generated_images, -1, 1)
                
                full = (hdr+1)/2
                inv_tone = True
                if inv_tone:
                    full_inv_tonemap = torch.pow(full/5, gamma)
                    img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy() 
                else:
                    img_hdr_np = full.permute(0, 2, 3, 1)[0].detach().cpu().numpy()


                imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)

            # save video
            if do_save_image:
                sequence_path=f'{paths_config.save_image_path}/*.png'
                sequences = sorted(glob.glob(f'{sequence_path}'))[150:]
                video_name=paths_config.save_video_path
                video = imageio.get_writer(f'{video_name}', mode='I', fps=25, codec='libx264', bitrate='16M')
                for filename in sequences:
                    img = imageio.imread(filename)
                    img_cat = np.concatenate([image_crop2pano, img], axis=1)      #img size (256, 512, 3)
                    video.append_data(img_cat)

                video.close()

    def train_with_tta(self, use_tta=True, adaptive_scale=True, num_augmentations=5):
        """
        TTA(Test-Time Augmentation)를 사용한 GAN Inversion + 동적 스케일 조절

        S2R-HDR 논문의 불확실성 기반 도메인 적응을 구현합니다.

        흐름:
        1. LFOV 이미지에 N개 증강 적용
        2. 각 증강된 LFOV에 대해 GAN Inversion 수행
        3. 각 w_pivot으로 파노라마 생성
        4. 파노라마들의 분산으로 불확실성 U(x) 계산
        5. 스케일 조절: scale1 = 1 - U(x), scale2 = 1 + U(x)
        6. 최종 파노라마 생성

        Args:
            use_tta: TTA 사용 여부 (False면 기본 train()과 동일)
            adaptive_scale: 불확실성 기반 스케일 조절 여부
            num_augmentations: TTA 증강 수 (기본값: 5)

        Note:
            - 기본 시나리오: Laval 데이터로 파인튜닝 → TTA 불필요
            - TTA는 새 도메인 추론 시 선택적으로 활성화
        """
        if not use_tta:
            # TTA 비활성화 시 기본 train() 호출
            return self.train()

        use_ball_holder = True
        is_128x256 = False

        # TTA 증강기 초기화
        tta_augmentor = TTAAugmentor(augmentations=[
            IdentityAug(),                      # 원본
            ExpAug(param=0.3),                  # 밝게
            ExpAug(param=-0.3),                 # 어둡게
            WBAug(gains=[1.05, 1.0, 0.95]),    # 따뜻한 톤
            FlipAug(horizontal=True),          # 수평 뒤집기
        ][:num_augmentations])

        print(f"[TTA] Augmentor initialized with {tta_augmentor.num_augmentations} augmentations")
        print(f"[TTA] Adaptive scale: {adaptive_scale}")

        for image_name in tqdm(self.data_loader):
            env = EnvironmentMap(256, 'latlong')
            image_crop2pano = crop2pano(env, image_name)
            print('image max:', image_crop2pano.max())
            image = 2*(image_crop2pano/255.0)-1

            image = torch.tensor(image.transpose([2, 0, 1]), device=global_config.device)
            image = image.unsqueeze(0).to(torch.float32)

            self.restart_training()
            name = os.path.splitext(os.path.basename(image_name))[0]

            # 마스크 및 bbox 설정
            if is_128x256:
                mask_fname = '/home/deep/projects/mini-stylegan2/crop60.jpg'
            else:
                mask_fname = 'crop60_256x512.jpg'
            mask_pil = PIL.Image.open(mask_fname).convert('RGB')

            if is_128x256:
                mask_pil = mask_pil.resize((256, 128), PIL.Image.LANCZOS)
            else:
                mask_pil = mask_pil.resize((512, 256), PIL.Image.LANCZOS)

            mask_pil_sum_c = np.sum(mask_pil, axis=2)
            mask_pil_sum_c_row = np.sum(mask_pil_sum_c, axis=1)
            mask_pil_sum_c_col = np.sum(mask_pil_sum_c, axis=0)
            row_min = np.argwhere(mask_pil_sum_c_row).min() + 10
            row_max = np.argwhere(mask_pil_sum_c_row).max() - 5
            col_min = np.argwhere(mask_pil_sum_c_col).min() + 10
            col_max = np.argwhere(mask_pil_sum_c_col).max() - 10

            bbox = [row_min, row_max, col_min, col_max]

            # ============================================================
            # TTA: 여러 증강된 LFOV로 불확실성 계산
            # ============================================================
            print(f"[TTA] Computing uncertainty for {name}...")

            augmented_inputs = tta_augmentor.generate_augmented_inputs(image)
            panoramas = []
            augmentations = []

            for aug_image, aug_module in augmented_inputs:
                # 각 증강된 이미지로 GAN Inversion
                w_pivot = self.calc_inversions(aug_image, name, bbox)
                w_pivot = w_pivot.to(global_config.device)

                # 파노라마 생성
                with torch.no_grad():
                    panorama = self.forward(w_pivot)
                    panoramas.append(panorama)
                    augmentations.append(aug_module)

            # 역변환 적용하여 정렬
            aligned_panoramas = tta_augmentor.apply_inverse_transforms(panoramas, augmentations)

            # 불확실성 계산
            uncertainty, variance_map = compute_uncertainty_from_outputs(
                aligned_panoramas,
                uncertainty_scale=0.05  # GT 없는 경우
            )

            print(f"[TTA] Uncertainty: {uncertainty.item():.6f}")

            # 적응형 스케일 적용
            if adaptive_scale:
                scale1, scale2 = apply_adaptive_scales(self.G, uncertainty)
                print(f"[TTA] Applied scales: scale1={scale1:.4f}, scale2={scale2:.4f}")
            else:
                scale1, scale2 = 1.0, 1.0

            # ============================================================
            # 최종 파노라마 생성 (원본 이미지 사용)
            # ============================================================
            w_pivot = self.calc_inversions(image, name, bbox)
            w_pivot = w_pivot.to(global_config.device)
            real_images_batch = image.to(global_config.device)
            real_images_batch = real_images_batch[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

            # PTI 최적화 (기존과 동일)
            for i in tqdm(range(hyperparameters.max_pti_steps), desc="PTI"):
                generated_images = self.forward(w_pivot)
                generated_images = torch.clip(generated_images, -1, 1)
                generated_images_cropped = generated_images[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

                loss, l2_loss_val, loss_lpips = self.calc_loss(
                    generated_images_cropped, real_images_batch, name,
                    self.G, use_ball_holder, w_pivot
                )

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
                global_config.training_step += 1

            self.image_counter += 1

            # 최종 출력 저장
            generated_images = self.forward(w_pivot)
            gamma = 2.4
            limited = True

            if limited:
                generated_images_singlemap = torch.mean(generated_images, dim=1, keepdim=True)
                r_percentile = torch.quantile(generated_images_singlemap, 0.999)
                light_mask = (generated_images_singlemap > r_percentile) * 1.0
                hdr = torch.clip(generated_images * (1 - light_mask), -1, 1) + \
                      torch.clip(generated_images * light_mask, -1, 2)
            else:
                hdr = torch.clip(generated_images, -1, 1)

            full = (hdr + 1) / 2
            full_inv_tonemap = torch.pow(full / 5, gamma)
            img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

            # 파일명에 TTA 정보 추가
            output_name = f'{name}_tta_s1{scale1:.2f}_s2{scale2:.2f}_u{uncertainty.item():.4f}'
            imsave(f'{paths_config.checkpoints_dir}/{output_name}.exr', img_hdr_np)
            print(f"[TTA] Saved: {output_name}.exr")






