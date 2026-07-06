import datasets
import pyarrow.parquet as pq 

def main():
    print("Hello from cart-segmentation-training!")
    # Step 0 : try to load and display samples from the dataset : 
    dataset = pq.read_table("cart_dataset/data.parquet")
    print(type(dataset))
    print(len(dataset))


if __name__ == "__main__":
    main()
