from llava.train.train_ce_new import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
