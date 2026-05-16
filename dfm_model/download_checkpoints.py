import subprocess

# Your exact RunPod storage volume configuration
ENDPOINT = "https://s3api-us-mo-1.runpod.io"
REGION = "us-mo-1"
BUCKET_PATH = "s3://bto7mb7v4l/NA-MPNN/dfm_model/"
LOCAL_DEST = "/Users/theoneandonly/Documents/Documents - Michael’s MacBook Pro (2)/University/Year 4/Part III Project/Model_pt/"

# The exact list of files you provided
FILES_TO_DOWNLOAD = [
    "dfm_log.jsonl", "dfm_log.txt", "last.pt", "log.txt", "s_10016.pt", 
    "s_10315.pt", "s_10514.pt", "s_10806.pt", "s_11005.pt", "s_1115.pt", 
    "s_11295.pt", "s_11587.pt", "s_11786.pt", "s_12068.pt", "s_12278.pt", 
    "s_12591.pt", "s_1264.pt", "s_12776.pt", "s_13054.pt", "s_13334.pt", 
    "s_13519.pt", "s_13824.pt", "s_14033.pt", "s_14331.pt", "s_14535.pt", 
    "s_14835.pt", "s_15032.pt", "s_15331.pt", "s_15534.pt", "s_1555.pt", 
    "s_15841.pt", "s_16044.pt", "s_16338.pt", "s_16535.pt", "s_16841.pt", 
    "s_17039.pt", "s_17333.pt", "s_17528.pt", "s_17819.pt", "s_18019.pt", 
    "s_18307.pt", "s_18507.pt", "s_1852.pt", "s_18805.pt", "s_19097.pt", 
    "s_19304.pt", "s_19508.pt", "s_19797.pt", "s_20085.pt", "s_20275.pt", 
    "s_20566.pt", "s_20760.pt", "s_21042.pt", "s_21252.pt", "s_2135.pt", 
    "s_21546.pt", "s_21829.pt", "s_22026.pt", "s_22320.pt", "s_22505.pt", 
    "s_22808.pt", "s_2287.pt", "s_23078.pt", "s_23278.pt", "s_23580.pt", 
    "s_23777.pt", "s_24085.pt", "s_24293.pt", "s_24574.pt", "s_24784.pt", 
    "s_25089.pt", "s_25289.pt", "s_25596.pt", "s_25791.pt", "s_2589.pt", 
    "s_26093.pt", "s_26284.pt", "s_26576.pt", "s_26774.pt", "s_27064.pt", 
    "s_27286.pt", "s_27541.pt", "s_27784.pt", "s_28037.pt", "s_282.pt", 
    "s_28296.pt", "s_28549.pt", "s_2863.pt", "s_28751.pt", "s_29068.pt", 
    "s_29263.pt", "s_29507.pt", "s_29757.pt", "s_30007.pt", "s_30254.pt", 
    "s_30532.pt", "s_30779.pt", "s_3091.pt", "s_31038.pt", "s_31301.pt", 
    "s_31547.pt", "s_31811.pt", "s_32062.pt", "s_32310.pt", "s_32546.pt", 
    "s_32792.pt", "s_3289.pt", "s_33043.pt", "s_33313.pt", "s_33555.pt", 
    "s_33800.pt", "s_34056.pt", "s_34260.pt", "s_34516.pt", "s_34771.pt", 
    "s_35026.pt", "s_35270.pt", "s_35549.pt", "s_35800.pt", "s_3589.pt", 
    "s_36059.pt", "s_36256.pt", "s_36501.pt", "s_36813.pt", "s_37062.pt", 
    "s_37251.pt", "s_37512.pt", "s_37775.pt", "s_3781.pt", "s_38026.pt", 
    "s_38272.pt", "s_38525.pt", "s_38769.pt", "s_39029.pt", "s_39280.pt", 
    "s_39280.pt", "s_39528.pt", "s_39767.pt", "s_40016.pt", "s_40267.pt", 
    "s_40528.pt", "s_40782.pt", "s_4085.pt", "s_41030.pt", "s_41251.pt", 
    "s_41516.pt", "s_41766.pt", "s_42019.pt", "s_42262.pt", "s_42519.pt", 
    "s_42775.pt", "s_4291.pt", "s_43027.pt", "s_43271.pt", "s_43525.pt", 
    "s_43778.pt", "s_44022.pt", "s_44261.pt", "s_44510.pt", "s_44762.pt", 
    "s_45018.pt", "s_45315.pt", "s_45551.pt", "s_45810.pt", "s_4586.pt", 
    "s_46067.pt", "s_46252.pt", "s_46505.pt", "s_46765.pt", "s_47018.pt", 
    "s_47277.pt", "s_47519.pt", "s_47787.pt", "s_4797.pt", "s_48049.pt", 
    "s_48290.pt", "s_48540.pt", "s_48775.pt", "s_49049.pt", "s_49311.pt", 
    "s_49516.pt", "s_49777.pt", "s_50021.pt", "s_50293.pt", "s_50562.pt", 
    "s_50755.pt", "s_5085.pt", "s_51012.pt", "s_51272.pt", "s_51512.pt", 
    "s_51770.pt", "s_52025.pt", "s_52275.pt", "s_52529.pt", "s_52787.pt", 
    "s_5284.pt", "s_53037.pt", "s_53285.pt", "s_53555.pt", "s_53802.pt", 
    "s_54004.pt", "s_54260.pt", "s_54512.pt", "s_54766.pt", "s_548.pt", 
    "s_55050.pt", "s_55316.pt", "s_55512.pt", "s_5569.pt", "s_55785.pt", 
    "s_56029.pt", "s_56300.pt", "s_56549.pt", "s_56818.pt", "s_57011.pt", 
    "s_57273.pt", "s_57533.pt", "s_5763.pt", "s_57793.pt", "s_58057.pt", 
    "s_58252.pt", "s_58510.pt", "s_58764.pt", "s_59017.pt", "s_59270.pt", 
    "s_59513.pt", "s_59761.pt", "s_60013.pt", "s_60275.pt", "s_60529.pt", 
    "s_6065.pt", "s_60789.pt", "s_61052.pt", "s_61252.pt", "s_61503.pt", 
    "s_61755.pt", "s_62053.pt", "s_62302.pt", "s_62552.pt", "s_6260.pt", 
    "s_62813.pt", "s_63002.pt", "s_63259.pt", "s_63505.pt", "s_63761.pt", 
    "s_64004.pt", "s_64263.pt", "s_64520.pt", "s_64765.pt", "s_65019.pt", 
    "s_65276.pt", "s_6551.pt", "s_65526.pt", "s_65773.pt", "s_66022.pt", 
    "s_66275.pt", "s_66544.pt", "s_66802.pt", "s_67003.pt", "s_67251.pt", 
    "s_67504.pt", "s_67806.pt", "s_68052.pt", "s_68293.pt", "s_6830.pt", 
    "s_68524.pt", "s_68777.pt", "s_69031.pt", "s_69302.pt", "s_69563.pt", 
    "s_69765.pt", "s_70011.pt", "s_70275.pt", "s_7028.pt", "s_7313.pt", 
    "s_7506.pt", "s_7774.pt", "s_8047.pt", "s_833.pt", "s_8331.pt", 
    "s_8530.pt", "s_8834.pt", "s_9027.pt", "s_9326.pt", "s_9528.pt", "s_9819.pt"
]

total_files = len(FILES_TO_DOWNLOAD)
print(f"Starting batch download of {total_files} files directly...")

for index, filename in enumerate(FILES_TO_DOWNLOAD, start=1):
    source_url = f"{BUCKET_PATH}{filename}"
    dest_path = f"{LOCAL_DEST}{filename}"
    
    # Build the precise S3 command for this exact file
    cmd = [
        "aws", "s3", "cp", 
        source_url, 
        dest_path, 
        "--region", REGION, 
        "--endpoint-url", ENDPOINT
    ]
    
    print(f"[{index}/{total_files}] Downloading {filename}...")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error downloading {filename}: {result.stderr}")

print("All done!")