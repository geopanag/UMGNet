import time
import argparse
import numpy as np

import torch

# import torch.nn.functional as F
import torch.optim as optim

from models.netdeconf import GCN_DECONF
import utils
import pandas as pd

# from scipy import sparse as sp
import csv
import os
from sklearn.model_selection import KFold


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--nocuda", type=int, default=0, help="Disables CUDA training.")
parser.add_argument("--dataset", type=str, default="Ours")
parser.add_argument("--extrastr", type=str, default="1")

parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-5,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=100, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.1, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0, help="trade-off of representation balancing."
)
parser.add_argument("--clip", type=float, default=100.0, help="gradient clipping")
parser.add_argument("--nout", type=int, default=2)
parser.add_argument("--nin", type=int, default=2)

parser.add_argument("--tr", type=float, default=0.6)
parser.add_argument("--path", type=str, default="./datasets/")
parser.add_argument("--normy", type=int, default=1)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

alpha = Tensor([args.alpha])

np.random.seed(args.seed)
torch.manual_seed(args.seed)

loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()
    loss = loss.cuda()
    bce_loss = bce_loss.cuda()


def eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp, col="avg_money_after"):
    model.eval()
    yf_pred, rep, p1 = model(X, A, T)  # p1 can be used as propensity scores
    # yf = torch.where(T>0, Y1, Y0)
    ycf_pred, _, _ = model(X, A, 1 - T)

    if args.dataset == "Ours":
        YF = torch.from_numpy(
            pd.read_csv(f"user_features_v4.csv", sep=",")[col].to_numpy()
        ).cuda()
    else:
        YF = torch.where(T > 0, Y1, Y0)
    YCF = torch.where(T > 0, Y0, Y1)

    ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
    # YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys

    y1_pred, y0_pred = torch.where(T > 0, yf_pred, ycf_pred), torch.where(
        T > 0, ycf_pred, yf_pred
    )

    if args.normy:
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
    pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test], (Y1 - Y0)[idx_test]))

    mae_ate_ts = torch.abs(
        torch.mean((y1_pred - y0_pred)[idx_test]) - torch.mean((Y1 - Y0)[idx_test])
    )
    print(
        "Test set results:",
        "pehe_ts= {:.4f}".format(pehe_ts.item()),
        "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()),
    )

    of_path = "./new_results/" + args.dataset + args.extrastr + "/" + str(args.tr)

    if args.lr != 1e-2:
        of_path += "lr" + str(args.lr)
    if args.hidden != 100:
        of_path += "hid" + str(args.hidden)
    if args.dropout != 0.5:
        of_path += "do" + str(args.dropout)
    if args.epochs != 50:
        of_path += "ep" + str(args.epochs)
    if args.weight_decay != 1e-5:
        of_path += "lbd" + str(args.weight_decay)
    if args.nout != 1:
        of_path += "nout" + str(args.nout)
    if args.alpha != 1e-5:
        of_path += "alp" + str(args.alpha)
    if args.normy == 1:
        of_path += "normy"

    of_path += ".csv"

    of = open(of_path, "a")
    wrt = csv.writer(of)
    wrt.writerow([pehe_ts.item(), mae_ate_ts.item()])


def prepare(i_exp, train_perc):
    # Load data and init models
    if args.dataset == "Ours":
        X, A, T = utils.load_data_ours(args.path, name=args.dataset)
        X = (Tensor(X[0]), Tensor(X[1]))
        T = LongTensor(np.squeeze(T))
        A = utils.sparse_mx_to_torch_sparse_tensor(A, cuda=args.cuda)

        n = X[0].shape[0]
        n_train = int(n * train_perc)
        n_test = int(n * 1 - train_perc)
        # n_valid = n_test

        idx = np.random.permutation(n)
        idx_train, idx_test, idx_val = (
            idx[:n_train],
            idx[n_train : n_train + n_test],
            idx[n_train + n_test :],
        )
        idx_train = LongTensor(idx_train)
        idx_val = LongTensor(idx_val)
        idx_test = LongTensor(idx_test)

        # print(X.shape, Y1.shape, A.shape)

        # Model and optimizer
        model = GCN_DECONF(
            nfeat=(X[0].shape[-1], X[1].shape[-1]),
            nhid=args.hidden,
            num_users=n,
            dropout=args.dropout,
            n_out=args.nout,
            n_in=args.nin,
            cuda=args.cuda,
        )

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        return X, A, T, idx_train, idx_val, idx_test, model, optimizer

    else:
        X, A, T, Y1, Y0 = utils.load_data(
            args.path,
            name=args.dataset,
            original_X=False,
            exp_id=str(i_exp),
            extra_str=args.extrastr,
        )
        n = X.shape[0]
        n_train = int(n * args.tr)
        n_test = int(n * 0.2)
        # n_valid = n_test

        idx = np.random.permutation(n)
        idx_train, idx_test, idx_val = (
            idx[:n_train],
            idx[n_train : n_train + n_test],
            idx[n_train + n_test :],
        )
        X = utils.normalize(X)  # row-normalize
        # A = utils.normalize(A+sp.eye(n))

        X = X.todense()
        X = Tensor(X)

        Y1 = Tensor(np.squeeze(Y1))
        Y0 = Tensor(np.squeeze(Y0))
        T = LongTensor(np.squeeze(T))

        A = utils.sparse_mx_to_torch_sparse_tensor(A, cuda=args.cuda)

        # print(X.shape, Y1.shape, A.shape)

        idx_train = LongTensor(idx_train)
        idx_val = LongTensor(idx_val)
        idx_test = LongTensor(idx_test)

        # Model and optimizer
        model = GCN_DECONF(
            nfeat=X.shape[1],
            nhid=args.hidden,
            num_users=n,
            dropout=args.dropout,
            n_out=args.nout,
            n_in=args.nin,
            cuda=args.cuda,
        )

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        return X, A, T, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer


def train(epoch, X, A, T, Y1, Y0, idx_train, idx_val, model, optimizer, outcome):
    t = time.time()
    model.train()
    #    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.zero_grad()
    yf_pred, rep, p1 = model(X, A, T)
    ycf_pred, _, p1 = model(X, A, 1 - T)

    # representation balancing, you can try different distance metrics such as MMD
    rep_t1, rep_t0 = (
        rep[idx_train][(T[idx_train] > 0).nonzero()],
        rep[idx_train][(T[idx_train] < 1).nonzero()],
    )

    if args.dataset == "Ours":
        if args.cuda:
            YF = (
                torch.from_numpy(
                    pd.read_csv(f"user_features_v4.csv", sep=",")[outcome].to_numpy()
                )
                .float()
                .cuda()
            )
        else:
            YF = torch.from_numpy(
                pd.read_csv(f"user_features_v4.csv", sep=",")[outcome].to_numpy()
            ).float()
    else:
        YF = torch.where(T > 0, Y1, Y0)
    # YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        # recover the normalized outcomes
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]

    loss_train = loss(yf_pred[idx_train], YFtr)

    if alpha > 0:
        dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)
        loss_train = loss_train + alpha * dist

    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # validation
        loss_val = loss(yf_pred[idx_val], YFva)
        if alpha > 0:
            loss_val = loss_val + alpha * dist

        y1_pred, y0_pred = torch.where(T > 0, yf_pred, ycf_pred), torch.where(
            T > 0, ycf_pred, yf_pred
        )

        # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
        if args.normy:
            y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

        # in fact, you are not supposed to do model selection w. pehe and mae_ate
        # but it is possible to calculate with ITE ground truth (which often isn't available)

        # pehe_val = torch.sqrt(loss((y1_pred - y0_pred)[idx_val],(Y1 - Y0)[idx_val]))
        # mae_ate_val = torch.abs(
        #     torch.mean((y1_pred - y0_pred)[idx_val])-torch.mean((Y1 - Y0)[idx_val]))

        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "loss_val: {:.4f}".format(loss_val.item()),
            # 'pehe_val: {:.4f}'.format(pehe_val.item()),
            # 'mae_ate_val: {:.4f}'.format(mae_ate_val.item()),
            "time: {:.4f}s".format(time.time() - t),
        )


def uplift_score(prediction, treatment, target, rate=0.2):
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score


if __name__ == "__main__":
    os.chdir("../../data/RetailHero")

    for outcome in ["avg_money_change", "avg_money_after"]:
        # if outcome=='avg_money_change':
        train_percs = [20, 5]
        for train_perc in train_percs:
            for i_exp in range(5):
                fw = open(
                    f"../results/netdeconf_result_{i_exp}_{outcome}_{train_perc}.txt",
                    "a",
                )
                print(i_exp)
                torch.manual_seed(i_exp)
                np.random.seed(i_exp)
                torch.cuda.empty_cache()

                # X, A, T, idx_train, idx_val, idx_test, model, optimizer = prepare(i_exp,train_perc)
                X, A, T = utils.load_data_ours(args.path, name=args.dataset)
                YF = pd.read_csv(f"user_features_v4.csv", sep=",")[outcome]
                X = (Tensor(X[0]), Tensor(X[1]))
                # print(X[0][1,:])
                T = LongTensor(np.squeeze(T))
                n = X[0].shape[0]
                A = utils.sparse_mx_to_torch_sparse_tensor(A, cuda=args.cuda)

                kf = KFold(n_splits=abs(train_perc), shuffle=True, random_state=i_exp)
                result_fold = []
                up20 = []
                up40 = []
                for train_indices, test_indices in kf.split(T):
                    idx_test, idx_train = train_indices, test_indices

                    idx_train = LongTensor(idx_train)
                    idx_val = LongTensor([])
                    idx_test = LongTensor(idx_test)

                    # Model and optimizer
                    model = GCN_DECONF(
                        nfeat=(X[0].shape[-1], X[1].shape[-1]),
                        nhid=args.hidden,
                        num_users=n,
                        dropout=args.dropout,
                        n_out=args.nout,
                        n_in=args.nin,
                        cuda=args.cuda,
                    )

                    optimizer = optim.Adam(
                        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                    )

                    for epoch in range(args.epochs):
                        train(
                            epoch,
                            X,
                            A,
                            T,
                            None,
                            None,
                            idx_train,
                            idx_val,
                            model,
                            optimizer,
                            outcome,
                        )
                    print("Optimization Finished!")
                    # yf, T = eva(X, A, T, None, None, idx_train, idx_test, model, i_exp)

                    model.eval()
                    yf_pred, rep, p1 = model(X, A, T)
                    ycf_pred, rep, p1 = model(X, A, 1 - T)

                    # ----- from eva function
                    y1_pred, y0_pred = torch.where(
                        T > 0, yf_pred, ycf_pred
                    ), torch.where(T > 0, ycf_pred, yf_pred)

                    ite = y1_pred - y0_pred

                    # --------
                    idx_test = idx_test.detach().cpu().numpy()
                    YF_ = YF.to_numpy()[idx_test]
                    T_ = (
                        T.detach().cpu().numpy()[idx_test]
                    )  # df["treatment_flg"].to_numpy()[idx_test]
                    ite = ite.detach().cpu().numpy()[idx_test]

                    # print(YF[T==1].mean()-YF[T==0].mean())
                    # print(yf_pred[T==1].mean()-yf_pred[T==0].mean())

                    up20.append(uplift_score(ite, T_, YF_, 0.2))
                    up40.append(uplift_score(ite, T_, YF_, 0.4))

                    print(
                        f"{outcome} ---------- {np.mean(up20)} ------  {np.mean(up40)}"
                    )
                    fw.write(f"{up20[-1] },{up40[-1]}\n")
                    fw.flush()
                # print(f'{outcome} ---------- {np.mean(up20)} ------  {np.mean(up40)}')
                # fw.write(f'{np.mean(up20) }_{np.mean(up40)}\n')
                fw.close()
        # Testing
        # if args.dataset == 'Ours':
        #    eva(X, A, T, None, None, idx_train, idx_test, model, i_exp)
        # else:
        #    eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp)
