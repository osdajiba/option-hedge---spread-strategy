# Derivative Hedge

Author: Jacky Lee

This project aims to provide tools for analyzing and constructing derivative spreads, such as bull spreads and bear spreads, using options contracts.

## Installation

1. Clone the repository:
  
   ```bash
   git clone https://github.com/your_username/derivative-hedge.git
   ```
  
2. Install the required dependencies:
  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Modify the configurations in `conf/config.json` according to your data file paths.  

2. <font color=blue>Check if you have the files `db/available_data_path` and `db/datasets`. If you do NOT have these files, please run the notebook file first: `db/data_engine.ipynb`</font>  
  
3. Run the main script if using windows cmd:
  
   ```bash
   ./bin/launch.bat
   ```
  
   **you can go to file `bin` , and then click `launch.bat` to launch the program, or using IDE like PyCharm or vscode to run `./bin/main.py`*
  
4. Finally，if you want the statical result, please run the notebook file : `bin/result.ipynb`, <font color=blue>which will generate file  `db/result` which is larger than 1GB</font>.
  
*<font color=red>*Running `bin/result.ipynb` will calling default processors = 16 ! Please check out if your computer's cores and logical processors are available before you run the program.*</font>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome! Please fork the repository and create a pull request.

## Contact

For any questions or feedback, please contact [2354889815@qq.com](mailto:your_email@example.com).
