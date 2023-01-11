import os
import shutil
from fe_function import * 

output_dir = './data/fe/'
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)

generate_customer_gender()
generate_hotness_articles()
generate_customer_features()
generate_dynamic_article_df()
generate_static_article_df()
generate_collaborative_filter()