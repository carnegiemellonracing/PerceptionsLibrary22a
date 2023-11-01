# 22a's Perceptions Library

## Setup

1. **Clone the Repository:**
   ```bash
   git clone [repository-link]
   cd PerceptionsLibrary22a
   ```

2. **Setup Virtual Environment:**
   Ensure you have Python 3.8 installed, then create a virtual environment:
   ```bash
   python3.8 -m venv env
   source env/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set PYTHONPATH:**
   To ensure `import perc22a` works in any script, add the absolute path of the `PerceptionsLibrary22a` to your `PYTHONPATH`:
   ```bash
   echo "export PYTHONPATH=\"$(pwd):$PYTHONPATH\"" >> ~/.zshrc # or ~/.bashrc
   source ~/.zshrc # or ~/.bashrc
   ```

5. **Verify Setup:**
   Confirm the path was correctly added by echoing the `$PYTHONPATH`:
   ```bash
   echo $PYTHONPATH
   ```
   Test the setup:
   ```bash
   python scripts/test_setup.py
   ```
   Successful output: `"Running 'import perc22a' successful"`.


## Loading Data

1. **Download Data:** 
   Fetch the data from [this Google Drive Link](https://drive.google.com/drive/folders/12l2DpvS4oEfl7_Noc7oUX4AcIDCfB8Zc?usp=drive_link) and place the `<name>.tar.gz` files in the `data/raw/` directory. Note: The files are large and can expand to more than 10GB when extracted.

2. **Extract Data:**
   ```bash
   tar -zxvf data/raw/<name>.tar.gz
   ```
   This creates a `data/raw/<name>` directory containing numerous `instance-<n>.npz` files, which represent snapshots of sensor data during track testing.

3. **Use DataLoader:**
   The `DataLoader` class, found in `data/utils/dataloader.py`, provides a convenient method for data access.
   
   To demonstrate its use:
   ```bash
   python3 scripts/load_data.py
   ```
   This displays a `cv2` window. Click on the image and press any key to navigate through the data. To exit, either hit `<Ctrl-C>` in the terminal and press a key in the `cv2` window or continue pressing keys until all images are cycled through.