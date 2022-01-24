from encoding import *

sys.path.append("vendor/FID_IS_infinity")
from score_infinity import calculate_FID_infinity_path

BATCH_SIZE = 4
TMP_DIR = "tmp/encoder_eval"
DATASET_PATH = "./datasets/afhq2_cat256_test.zip"
STATS_PATH = "datasets/afhq2_cat256_stats.npz"
MIN_FAKE = 100

if __name__ == "__main__":
    E = open_encoder("encoder-training-runs/encoder_0.0/2022-01-05_12:03:21/encoder-snapshot-000100.pkl")
    E.eval()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR)

    criterion = VGGCriterion()
    loader = EncodingDataset(path=DATASET_PATH).to_loader(
        batch_size=BATCH_SIZE, infinite=False
    )

    distances = []

    def compute_distance():
        return torch.stack(distances).mean()

    with torch.no_grad():
        progress = tqdm.tqdm(loader)
        for i, batch in enumerate(progress):
            batch = batch.cuda()
            preds = E(batch).to_image()
            distances.append(criterion(preds, batch).mean().cpu())

            for j, image in enumerate(preds):
                save_image(image, f"tmp/encoder_eval/{i}_{j}.png")

            if i % 3 == 0:
                progress.set_description(str(compute_distance().item()))
    
    print("Calculating FID...")
    fid = calculate_FID_infinity_path(
            STATS_PATH, TMP_DIR, BATCH_SIZE, min_fake=MIN_FAKE
        )
    
    print("Dist", compute_distance().item())
    print("FID", fid)
