import argparse
import plots




def get_plot_argparser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_class", choices=model_class_dict.keys())
    parser.add_argument("--model_save_file")




def get_model(args):
    return load_generative_model(args.model_class, args.model_save_file)