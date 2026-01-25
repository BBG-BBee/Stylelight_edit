최종 통합 프로젝트 마일스톤: 물리적 정합성 기반 HDR 뷰 확장 및 DGP 시뮬레이션
1. 프로젝트 정의 및 목표
본 프로젝트는 제한된 시야(NFoV, 약 63°)를 가진 단일 HDR 이미지(23mm 렌즈 등)를 입력받아, 물리적으로 정확한 $180^\circ$ 반구형(Hemispherical) HDR 파노라마를 생성하는 것을 목표로 한다.
●	핵심 목표: 단순한 시각적 확장(Outpainting)을 넘어, 주광 눈부심 확률(DGP) 계산의 정확성을 담보할 수 있는 수직 조도($E_v$) 정합성을 확보한다1.
●	해결 과제: 기존 모델들이 간과했던 **전이 구간($300 \sim 1,000 cd/m^2$)**의 휘도 정보를 정밀하게 복원하여, 시뮬레이션 시 발생하는 '거짓 양성(False Positive)' 눈부심 오류를 제거한다2222.
________________________________________
2. 핵심 R&D 전략 (Core Strategy)
2.1 모델 아키텍처: HDR-Native StyleGAN2
●	기반 모델: StyleLight (GAN) 기반이나, LDM의 VAE 정보 손실(Blurring)을 피하기 위해 선택됨3.
●	활성화 함수 변경: Generator의 최종 출력층을 Tanh($-1 \sim 1$)에서 Softplus($0 \sim \infty$)로 교체하여, 물리적 휘도의 비음수성(Non-negativity)과 무한한 동적 범위를 지원4.
●	해상도: $512 \times 1024$ (Equirectangular) Native 출력5.
2.2 학습 전략: 2-Stage & S2R-Adapter
●	Stage 1 (구조 학습): 대규모 합성 데이터(S2R-HDR)를 사용하여 일반적인 기하학적 구조 확장 능력 배양. 단, 합성 데이터의 물리적 부정확성을 고려하여 DTAM은 비활성화6.
●	Stage 2 (물리 보정): 실측 데이터(Laval Photometric)와 **S2R-Adapter (3-브랜치 도메인 적응 구조)**를 사용하여 물리적 도메인 적응. 이때 DTAM을 활성화하여 $E_v$ 정합성 확보7777.
2.3 핵심 기술: DTAM 및 Full FP32
●	DTAM (이중 임계값 적응형 마스킹): 휘도에 따라 가중치를 차등 부여하는 손실 함수 전략.
○	$T_{onset} = 300 cd/m^2$ (학습 시작점), $T_{peak} = 1,000 cd/m^2$ (최대 가중치 도달점)8.
●	Full FP32 Enforcement: NVIDIA RTX 5090 (32GB+ VRAM) 환경에서 모든 연산(데이터 로더, 가중치, 손실)을 float32로 강제하여 Overflow/Underflow 방지999.
________________________________________
3. 단계별 세부 마일스톤 (Detailed Milestones)
Milestone 1: 인프라 구축 및 데이터 파이프라인 (기반 조성)
●	목표: 물리적 정합성을 지원하는 하드웨어 및 소프트웨어 환경 구축.
●	세부 활동:
1.	하드웨어 확정: NVIDIA RTX 5090 확보 및 CUDA 환경 설정 (Full FP32 메모리 할당 테스트)10.
2.	데이터셋 준비:
■	Stage 1용: S2R-HDR (24k장, 합성)11.
■	Stage 2용: Laval Photometric Indoor HDR (1.7k장, $cd/m^2$ 보정)12.
3.	데이터 로더 개발: pytorch360convert를 커스터마이징하여 톤매핑(Tone-mapping)이나 감마 보정 없이 선형(Linear) Float32 텐서를 직접 로드하는 파이프라인 구축13.
Milestone 2: 모델 아키텍처 수정 및 초기화
●	목표: HDR 데이터를 손실 없이 처리할 수 있는 모델 구조 변경.
●	세부 활동:
1.	출력층 교체: StyleGAN2 Generator의 마지막 ToRGB 레이어 활성화 함수를 Softplus로 변경14.
2.	FP32 강제화: 모델 코드 내 amp (Automatic Mixed Precision) 관련 코드 전면 제거 및 torch.set_default_dtype(torch.float32) 적용15.
3.	판별자($D$) 수정: $180^\circ$ HDR 이미지를 입력받아 리얼리티를 판별하는 $D_{180\_HDR}$ 구현.
Milestone 3: Stage 1 파인튜닝 (구조 및 뷰 확장 학습)
●	목표: S2R-HDR 데이터셋을 활용하여 '잘린 이미지를 자연스럽게 확장'하는 능력 확보.
●	주요 전략:
○	데이터: S2R-HDR.
○	마스킹: DTAM 비활성화 (OFF). (합성 데이터의 광원 물리량이 실제와 다를 수 있으므로 구조 학습에만 집중) 16.
○	손실 함수: 기본 GAN 손실 + 지각적 손실(LPIPS).
●	결과물: Checkpoint-Stage1 (구조적 이해도가 높은 기본 모델).
Milestone 4: Stage 2 파인튜닝 (물리적 정합성 및 DTAM 적용)
●	목표: Laval 실측 데이터를 통해 실제 빛의 세기와 분포($E_v$)를 학습.
●	주요 전략:
○	데이터: Laval Photometric Indoor HDR.
○	S2R-Adapter: Stage 1의 Generator($G$) 가중치를 **동결(Freeze)**하고, 3-브랜치 구조(shared + transfer1 + transfer2)의 어댑터만 학습. r1=1(미세 조정), r2=128(광범위 적응)으로 구성17.
○	DTAM 활성화 (ON): 배치 내 각 이미지에 대해 동적 마스크 $W$ 생성 ($T_{onset}=300, T_{peak}=1000$ 적용)18.
○	손실 함수:
$$\mathcal{L}_{Total} = \mathcal{L}_{Phys}(DTAM) + \lambda \mathcal{L}_{Consist} + \mathcal{L}_{GAN}$$
■	$\mathcal{L}_{Phys}$: DTAM 가중치가 적용된 물리적 L1 손실.
■	$\mathcal{L}_{Consist}$: Stage 1 모델과의 구조적 차이를 제한하는 일관성 손실19.
●	결과물: Checkpoint-Final (물리적으로 보정된 최종 모델).
Milestone 5: 검증 파이프라인 및 추론 (Inference Workflow)
●	목표: 해상도 손실 없이 evalglare 분석이 가능한 고품질 결과물 생성.
●	실행 프로세스 (8-Step Workflow)20:
1.	생성: $512 \times 1024$ Equirectangular HDR 생성 (FP32).
2.	크롭: 전방 $180^\circ$ 영역 추출 ($512 \times 512$).
3.	초해상도 (SR): 작은 광원 뭉개짐 방지를 위해 $1024 \times 1024$로 업스케일링 (SwinIR 등 활용).
4.	변환: Angular Fisheye (-vta) 포맷으로 투영 변환.
5.	헤더 주입: Radiance 헤더(VIEW= -vta -vv 180 -vh 180) 삽입.
6.	DGP 산출: evalglare 실행.
Milestone 6: 최종 평가 (Evaluation Metrics)
●	정량적 평가:
○	$\Delta E_v$ (수직 조도 오차율): 목표 < 10%. (가장 중요한 지표) 21.
○	$RMSE_{trans}$ (전이 구간 오차): $300 \sim 1,000 cd/m^2$ 픽셀들에 대한 복원 정확도22.
○	DGP Class Accuracy: 눈부심 등급(Imperceptible ~ Intolerable) 분류 정확도23.
○	기존 지표: PU21-PSNR, HDR-VDP-3 Q-Score24.
●	정성적 평가:
○	Blender를 이용한 IBL(Image-Based Lighting) 렌더링 테스트 (반사/그림자 품질 비교)25.

