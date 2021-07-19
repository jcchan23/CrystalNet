from chemprop.parsing import parse_predict_args, modify_predict_args
from chemprop.train import make_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    test_name, test_prediction = make_predictions(args)

    with open(f'{args.test_path}/seed_{args.seed}/predict_{args.dataset_name}_crystalnet.csv', 'w') as fw:
        fw.write(f'name,{args.dataset_name}\n')

        for name, prediction in zip(test_name, test_prediction):
            fw.write(f'{name},{",".join([str(predict) for predict in prediction])}\n')
