import json
import torch
import os
from model_scripts.train_model_ae import *
from evaluation_scripts.evaluation_helper import *
from model_scripts.clustering import *
import pickle
import random
import copy


def create_model(model_class, channels=10, out_channels=10, timestamps=7, latent=32, subpatch_size=4, non_temp=False, time_emb_channel= False):
    if time_emb_channel == True:
        return model_class(12, out_channels, timestamps, latent, subpatch_size)
    if non_temp == True:
        return model_class(channels, latent, subpatch_size)
    return model_class(channels, timestamps, latent, subpatch_size)


def train_model_multiple_runs_with_metrics(
    model_name, model_class, dataloader_train, dataloader_test, dataloader_eval, channels, timestamps,
    epochs, optimizer, lr, vae_lr, vae_optimizer, momentum, device, config, subpatch_size=config.subpatch_size, output_dir="results", non_temp=False
):
    """Run 3 executions of training-evaluation for every model"""

    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = os.path.join(output_dir, "Trained_Models")
    os.makedirs(model_save_dir, exist_ok=True)

    all_train_losses = []
    all_test_losses = []
    all_train_kl_losses = []
    all_test_kl_losses = []
    all_metrics = []

    best_recall = -1
    best_model = None

    for run in range(3):
        # model = copy.deepcopy(model_obj)           # fresh copy
        if model_name in ['3D_AE_temporal_channel']:
            model = create_model(model_class, channels=channels, timestamps=timestamps, time_emb_channel=True) #for 2 extra encoding channels
        elif model_name in ['3D_AE_16', '3D_AE_8']:
            model = create_model(model_class, channels=channels, timestamps=timestamps, subpatch_size=subpatch_size) #subpatch size different
        else:
            model = create_model(model_class, channels=channels, timestamps=timestamps, non_temp=non_temp, time_emb_channel=False)   #no extra channels
        model = model.to(device)

        if model_name in ['2D_AE','3D_AE','2D_AE_NonTemporal']:
            trained_model, train_losses, test_losses = train_model_ae(
                model, dataloader_train, dataloader_test,
                epochs=epochs, optimizer=optimizer, lr=lr,
                momentum=momentum, device=device
            )
        if model_name in ['3D_AE_temporal_channel']:
            trained_model, train_losses, test_losses = train_model_ae_te(
                model, dataloader_train, dataloader_test,
                epochs=epochs, optimizer=optimizer, lr=lr,
                momentum=momentum, device=device
            )
        if model_name in ['3D_AE_temporal_addition', '3D_AE_MVI', '3D_AE_B4', '3D_AE_16', '3D_AE_8', '3D_AE_temporal_addition_2024','2D_AE_temporal_addition']:
            trained_model, train_losses, test_losses = train_model_ae_te_pixel(
                model, dataloader_train, dataloader_test,
                epochs=epochs, optimizer=optimizer, lr=lr,
                momentum=momentum, device=device
            )
        if model_name in ['3D_VAE','2D_VAE']:
            trained_model, train_recon_losses, train_kl_losses, test_recon_losses, test_kl_losses = train_model_vae(
                model, dataloader_train, dataloader_test,
                epochs=epochs, optimizer=vae_optimizer, lr=vae_lr,
                device=device
            )
            train_losses = train_recon_losses
            test_losses = test_recon_losses
            all_train_kl_losses.append(train_kl_losses)
            all_test_kl_losses.append(test_kl_losses)

        # Feature extraction
        if model_name in ['3D_VAE','2D_VAE']:
            train_features, train_coord_dl = extract_features_vae(trained_model, dataloader_train, device=device)
            test_features, test_coord_dl = extract_features_vae(trained_model, dataloader_test, device=device)
            eval_features, eval_coord_dl = extract_features_vae(trained_model, dataloader_eval, device=device)
        elif model_name in ['3D_AE_temporal_addition', '3D_AE_MVI', '3D_AE_B4', '3D_AE_16', '3D_AE_8', '3D_AE_temporal_addition_2024','2D_AE_temporal_addition']:
            train_features, train_coord_dl = extract_features_ae(trained_model, dataloader_train, temp_embed_pixel=True, device=device)
            test_features, test_coord_dl = extract_features_ae(trained_model, dataloader_test, temp_embed_pixel=True, device=device)
            eval_features, eval_coord_dl = extract_features_ae(trained_model, dataloader_eval, temp_embed_pixel=True, device=device)
        else:
            train_features, train_coord_dl = extract_features_ae(trained_model, dataloader_train, temp_embed_pixel=False, device=device)
            test_features, test_coord_dl = extract_features_ae(trained_model, dataloader_test, temp_embed_pixel=False, device=device)
            eval_features, eval_coord_dl = extract_features_ae(trained_model, dataloader_eval, temp_embed_pixel=False, device=device)


        train_features = train_features.cpu()
        test_features = test_features.cpu()
        eval_features = eval_features.cpu()

        combined_train_features = torch.cat((train_features, test_features), dim=0)
        combined_train_coords = train_coord_dl + test_coord_dl

        # Clustering
        kmeans = kmeans_function(combined_train_features, n_clusters=2, random_state=random.randint(1, 100))
        eval_preds = kmeans.predict(eval_features.reshape(eval_features.size(0), -1).numpy().astype(np.float32))

        # Evaluation
        disease, acc, precision, recall, f1_score, f2_score = evaluate_clustering_metrics(
            eval_coord_dl, eval_preds, config.labels_path, config.subpatch_to_patch_threshold
        )

        # Save best model (by recall)
        if recall > best_recall:
            best_recall = recall
            best_model = trained_model
            model_path = os.path.join(model_save_dir, f"{model_name}_best_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)

        # Save run data
        run_result = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "f2_score": f2_score
        }
        if model_name in ['3D_VAE','2D_VAE']:
            run_result["train_kl_losses"] = train_kl_losses
            run_result["test_kl_losses"] = test_kl_losses

        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_metrics.append({k: run_result[k] for k in ["accuracy", "precision", "recall", "f1_score", "f2_score"]})
        with open(f"{output_dir}/{model_name}_run{run+1}.json", "w") as f:
            json.dump(run_result, f)

    # Averages
    avg_train_loss = np.mean(all_train_losses, axis=0).tolist()
    avg_test_loss = np.mean(all_test_losses, axis=0).tolist()
    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in all_metrics[0]
    }

    with open(f"{output_dir}/{model_name}_avg.json", "w") as f:
        json.dump({
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            **{f"avg_{k}": v for k, v in avg_metrics.items()}
        }, f)

    return avg_train_loss, avg_test_loss, avg_metrics



def compile_results_table_with_metrics(model_names, output_dir="Results"):
    loss_rows = []
    acc_rows = []
    recall_rows = []
    precision_rows = []
    f1_rows = []

    for model_name in model_names:
        # === LOSS ROW ===
        loss_row = {"Model": model_name}
        acc_row = {"Model": model_name}
        recall_row = {"Model": model_name}
        precision_row = {"Model": model_name}
        f1_row = {"Model":model_name}

        for run in range(3):
            run_file = os.path.join(output_dir, f"{model_name}_run{run+1}.json")
            with open(run_file, "r") as f:
                data = json.load(f)

            # 50th epoch loss (index 49 if 0-indexed)
            loss_row[f"Loss Run {run+1}"] = data["test_losses"][49]
            acc_row[f"Accuracy Run {run+1}"] = data["accuracy"]
            recall_row[f"Recall Run {run+1}"] = data["recall"]
            precision_row[f"Precision Run {run+1}"] = data["precision"]
            f1_row[f"F1 Run {run+1}"] = data["f1_score"]

        # Load average file
        avg_file = os.path.join(output_dir, f"{model_name}_avg.json")
        with open(avg_file, "r") as f:
            avg_data = json.load(f)

        loss_row["Loss Avg"] = avg_data["avg_test_loss"][49]
        acc_row["Accuracy Avg"] = avg_data["avg_accuracy"]
        recall_row["Recall Avg"] = avg_data["avg_recall"]
        precision_row["Precision Avg"] = avg_data["avg_precision"]
        f1_row["F1-score Avg"] = avg_data["avg_f1_score"]

        loss_rows.append(loss_row)
        acc_rows.append(acc_row)
        recall_rows.append(recall_row)
        precision_rows.append(precision_row)
        f1_rows.append(f1_row)

    # Convert to DataFrames
    df_loss = pd.DataFrame(loss_rows)
    df_accuracy = pd.DataFrame(acc_rows)
    df_recall = pd.DataFrame(recall_rows)
    df_precision = pd.DataFrame(precision_rows)
    df_f1 = pd.DataFrame(f1_rows)

    return df_loss, df_accuracy, df_recall, df_f1, df_precision

