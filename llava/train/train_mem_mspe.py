from llava.train.train_mspe import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
