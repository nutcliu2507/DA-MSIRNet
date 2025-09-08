import subprocess

name = "Places2"
for i in range(4):
    mask_r = (i + 1) * 10
    mask_r_10 = mask_r + 10
    strmask = str(mask_r) + "-" + str(mask_r_10)
    mask_file = "datasets/mask/test_mask/" + strmask
    results_dir = r"C:\Users\2507\Desktop\spa-former-main\checkpoints\predict_" + name + "/" + strmask
    # dataset_path
    CelebAHQ_path = r"C:\Users\2507\Desktop\dataset\celebA-HQ\visual_test_source_256"
    Places2_path = r"C:\Users\2507\Desktop\dataset\places2\test_256"
    Paris_path = r"C:\Users\2507\Desktop\dataset\Paris\val"
    FFHQ_path = r"C:\Users\2507\Desktop\dataset\FFHQ\images\test"
    # # CelebA-HQ
    arguments = ["--name", name, "--checkpoints_dir", "./checkpoints", "--which_iter", "latest", "--gpu_ids", "0",
                 "--mask_type", "[3]", "--img_file", CelebAHQ_path, "--mask_file", mask_file, "--batchSize", "4",
                 "--how_many", "0", "--results_dir", results_dir, "--no_shuffle"]

    # # Places2
    # arguments = ["--name", name, "--checkpoints_dir", "./checkpoints", "--which_iter", "latest", "--gpu_ids", "0",
    #              "--mask_type", "[3]", "--img_file", Places2_path, "--mask_file", mask_file, "--batchSize", "4",
    #              "--how_many", "1000 ", "--results_dir", results_dir, "--no_shuffle"]

    # # Paris
    # arguments = ["--name", name, "--checkpoints_dir", "./checkpoints", "--which_iter", "latest", "--gpu_ids", "0",
    #              "--mask_type", "[3]", "--img_file", Paris_path, "--mask_file", mask_file, "--batchSize", "4",
    #              "--how_many", "0 ","--results_dir", results_dir, "--no_shuffle"]

    # # FFHQ
    # arguments = ["--name", name, "--checkpoints_dir", "./checkpoints", "--which_iter", "latest", "--gpu_ids", "0",
    #              "--mask_type", "[3]", "--img_file", FFHQ_path, "--mask_file", mask_file, "--batchSize", "4",
    #              "--how_many", "0 ","--results_dir", results_dir, "--no_shuffle"]

    # # MISTO(Use Places2 Weight)
    # arguments = ["--name", name, "--checkpoints_dir", "./checkpoints", "--which_iter", "latest", "--gpu_ids", "0",
    #              "--mask_type", "[3]", "--img_file", r"C:\Users\2507\Desktop\dataset\MISTO\512\image", "--mask_file",
    #              r"C:\Users\2507\Desktop\dataset\MISTO\512\mask", "--batchSize", "4", "--how_many", "0 ", "--results_dir",
    #              ./checkpoints/predict_MISATO", "--no_shuffle"]

    result = subprocess.run([r"C:\Users\2507\Envs\t181\Scripts\python.exe", "test.py"] + arguments, check=True)
