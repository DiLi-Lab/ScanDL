import sys
import os
import argparse
import datetime
import time
sys.path.append('.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--noise_schedule', type=str, default='sqrt',
                        choices=['linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin'],
                        help='the distribution of noises')
    parser.add_argument('--diff_steps', type=int, default=2000, help='diffusion steps')
    parser.add_argument('--schedule_sampler', type=str, default='lossaware', choices=['uniform', 'lossaware', 'fixstep'],
                        help='schedule sampler of timesteps')

    parser.add_argument('--seq_len', type=int, default=128, help='max len of input sequence')
    parser.add_argument('--hidden_t_dim', type=int, default=128, help='hidden size of time embedding')
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden size of word embedding and transformer hidden size')
    parser.add_argument('--learning_steps', type=int, default=60000, help='total steps of learning')
    parser.add_argument('--save_interval', type=int, default=2000, help='save step')
    parser.add_argument('--resume_checkpoint', type=str, default='none',
                        help='path to resume checkpoint, like xxx/xxx.pt')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--microbatch', type=int, default=64, help='microbatch size')
    parser.add_argument('--seed', type=int, default=101, help='random seed')

    parser.add_argument('--config_name', type=str, default='bert-base-cased', help='config of pre-trained models')
    parser.add_argument('--vocab', type=str, default='bert',
                        help='use bert vocab or load external vocab dict if given as path')
    parser.add_argument('--use_plm_init', type=str, default='no', choices=['no', 'bert'],
                        help='load init parameter from the pre-trained lm')
    parser.add_argument('--log_interval', type=int, default=200, required=False)
    parser.add_argument('--eval_interval', type=int, default=500, required=False)

    parser.add_argument('--notes', type=str, default='-', help='as training notes or specifical args', required=False)
    parser.add_argument('--app', type=str, default='', help='other input args')

    # further arguments
    parser.add_argument('--data_split_criterion', type=str, help='how to split the data into train, val, test:'
                        ' scanpath (random), reader, sentence, combined', required=False, default='reader')
    parser.add_argument('--num_transformer_layers', type=int, default=4, required=False, help='the number of encoder layers')
    parser.add_argument('--num_transformer_heads', type=int, default=8, required=False, help='the number of attention heads')
    parser.add_argument('--celer_only_L1', required=False, action='store_true', help='if given, all celer speakers are used'
                                                                                     'as opposed to only L1 speakers')
    parser.add_argument('--corpus', type=str, help='the eye-tracking corpus to use for training.', required=False,
                        default='celer', choices=['celer', 'zuco'])
    parser.add_argument('--inference', required=False, default='cv', choices=['cv', 'zuco', 'in-corpus'],
                        help='if zuco, inference is performed on zuco while trained on celer; if cv, inference is'
                             'done in k-fold Cross-Validation; if in-corpus, the training corpus is simply split into'
                             'train and test.')
    parser.add_argument('--mask_padding', action='store_false', required=False,
                        help='if given, padding will not be masked in transformer attention. if not given, mask_padding'
                             'is stored as True; padding will be masked.')
    parser.add_argument('--load_train_data', type=str, default='-',
                        help='if given, previously saved train data is loaded from the specified checkpoint path')

    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    folder_name = 'checkpoint-path'

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    today = datetime.date.today().strftime('%Y-%m-%d')

    if args.notes != '-':
        model_file = f'{today}_{args.notes}_split-{args.data_split_criterion}_h{args.hidden_dim}_seq{args.seq_len}_t{args.diff_steps}_{args.schedule_sampler}_' \
                     f'{args.noise_schedule}_ah{args.num_transformer_heads}_l{args.num_transformer_layers}_lr{args.lr}_' \
                     f'seed{args.seed}'
    else:
        model_file = f'{today}_split-{args.data_split_criterion}_h{args.hidden_dim}_seq{args.seq_len}_t{args.diff_steps}_{args.schedule_sampler}_' \
                     f'{args.noise_schedule}_ah{args.num_transformer_heads}_l{args.num_transformer_layers}_lr{args.lr}_' \
                     f'seed{args.seed}'

    model_file = os.path.join(folder_name, model_file)

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.exists(model_file):
            os.makedirs(model_file)

    COMMANDLINE = f'TOKENIZERS_PARALLELISM=FALSE ' \
                  f'python sp_train.py ' \
                  f'--checkpoint_path {model_file} ' \
                  f'--vocab {args.vocab} ' \
                  f'--use_plm_init {args.use_plm_init} ' \
                  f'--lr {args.lr} ' \
                  f'--batch_size {args.bsz} ' \
                  f'--microbatch {args.microbatch} ' \
                  f'--diffusion_steps {args.diff_steps} ' \
                  f'--noise_schedule {args.noise_schedule} ' \
                  f'--schedule_sampler {args.schedule_sampler} ' \
                  f'--seq_len {args.seq_len} ' \
                  f'--resume_checkpoint {args.resume_checkpoint} ' \
                  f'--hidden_t_dim {args.hidden_t_dim} ' \
                  f'--seed {args.seed} ' \
                  f'--hidden_dim {args.hidden_dim} ' \
                  f'--learning_steps {args.learning_steps} ' \
                  f'--save_interval {args.save_interval} ' \
                  f'--config_name {args.config_name} ' \
                  f'--notes {args.notes} ' \
                  f'--data_split_criterion {args.data_split_criterion} ' \
                  f'--num_transformer_layers {args.num_transformer_layers} ' \
                  f'--num_transformer_heads {args.num_transformer_heads} ' \
                  f'--corpus {args.corpus} ' \
                  f'--inference {args.inference} ' \
                  f'--load_train_data {args.load_train_data}' \

    COMMANDLINE += ' ' + args.app

    if int(os.environ['LOCAL_RANK']) == 0:
        with open(os.path.join(model_file, 'saved_bash.sh'), 'w') as f:
            print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    os.system(COMMANDLINE)