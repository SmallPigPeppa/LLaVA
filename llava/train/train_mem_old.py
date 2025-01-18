from llava.train.train_old import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
