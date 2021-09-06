from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys
from time import sleep

from PIL import Image
import io
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append('./indad')
from indad.data import MVTecDataset, StreamingDataset
from indad.models import SPADE, PaDiM, PatchCore
from indad.data import IMAGENET_MEAN, IMAGENET_STD

N_IMAGE_GALLERY = 4
N_PREDICTIONS = 2
METHODS = ["SPADE", "PaDiM", "PatchCore"]
BACKBONES = ["efficientnet_b0", "tf_mobilenetv3_small_100"]

# keep the two smallest datasets
mvtec_classes = ["hazelnut_reduced", "transistor_reduced"]

def tensor_to_img(x, normalize=False):
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x =  x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x

def pred_to_img(x, range):
    range_min, range_max = range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= (range_max - range_min)
    return tensor_to_img(x)

def show_pred(sample, score, fmap, range):
    sample_img = tensor_to_img(sample, normalize=True)
    fmap_img = pred_to_img(fmap, range)

    # overlay
    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    overlay_img = Image.open(buf)

    # actual display
    cols = st.columns(3)
    cols[0].subheader("Test sample")
    cols[0].image(sample_img)
    cols[1].subheader("Anomaly map")
    cols[1].image(fmap_img)
    cols[2].subheader("Overlay")
    cols[2].image(overlay_img)

def get_sample_images(dataset, n):
    n_data = len(dataset)
    ans = []
    if n < n_data:
        indexes = np.random.choice(n_data, n, replace=False)
    else:
        indexes = list(range(n_data))
    for index in indexes:
        sample, _ = dataset[index]
        ans.append(tensor_to_img(sample, normalize=True))
    return ans  

def main():
    with open("./docs/streamlit_instructions.md","r") as file:
        md_file = file.read()
    st.markdown(md_file)

    st.sidebar.title("Config")

    app_custom_dataset = st.sidebar.checkbox("Custom dataset", False)
    if app_custom_dataset:
        app_custom_train_images = st.sidebar.file_uploader(
            "Select 3 or more TRAINING images.",
            accept_multiple_files=True
        )
        app_custom_test_images = st.sidebar.file_uploader(
            "Select 1 or more TEST images.",
            accept_multiple_files=True
        )
        # null other elements
        app_mvtec_dataset = None
    else:
        app_mvtec_dataset = st.sidebar.selectbox("Choose an MVTec dataset", mvtec_classes)
        # null other elements
        app_custom_train_images = []
        app_custom_test_images = None

    app_method = st.sidebar.selectbox("Choose a method",
        METHODS)

    app_backbone = st.sidebar.selectbox("Choose a backbone",
        BACKBONES)

    manualRange = st.sidebar.checkbox('Manually set color range', value=False) 
    
    if manualRange:
        app_color_min = st.sidebar.number_input("set color min ",-1000,1000, 0)
        app_color_max = st.sidebar.number_input("set color max ",-1000,1000, 200)   
        color_range = app_color_min, app_color_max

    app_start = st.sidebar.button("Start")

    if app_start or "reached_test_phase" not in st.session_state:
        st.session_state.train_dataset = None
        st.session_state.test_dataset = None
        st.session_state.sample_images = None
        st.session_state.model = None
        st.session_state.reached_test_phase = False
        st.session_state.test_idx = 0
        # test_cols = None

    if app_start or st.session_state.reached_test_phase:
        # LOAD DATA
        # ---------
        if not st.session_state.reached_test_phase:
            flag_data_ok = False
            if app_custom_dataset:
                if len(app_custom_train_images) > 2 and \
                len(app_custom_test_images) > 0:
                    # test dataset will contain 1 test image
                    train_dataset = StreamingDataset()
                    test_dataset = StreamingDataset()
                    # train images
                    for training_image in app_custom_train_images:
                        bytes_data = training_image.getvalue()
                        train_dataset.add_pil_image(
                            Image.open(io.BytesIO(bytes_data))
                        )
                    # test image
                    for test_image in app_custom_test_images:
                        bytes_data = test_image.getvalue()
                        test_dataset.add_pil_image(
                            Image.open(io.BytesIO(bytes_data))
                        )
                    flag_data_ok = True
                else:
                    st.error("Please upload 3 or more training images and 1 test image.")
            else:
                with st_stdout("info", "Checking or downloading dataset ..."):
                    train_dataset, test_dataset = MVTecDataset(app_mvtec_dataset).get_datasets()
                    st.success(f"Loaded '{app_mvtec_dataset}' dataset.")
                    flag_data_ok = True
            
            if not flag_data_ok:
                st.stop()
        else:
            train_dataset = st.session_state.train_dataset
            test_dataset = st.session_state.test_dataset

        st.header("Random (healthy) training samples")
        cols = st.columns(N_IMAGE_GALLERY)
        if not st.session_state.reached_test_phase:
            col_imgs = get_sample_images(train_dataset, N_IMAGE_GALLERY)
        else:
            col_imgs = st.session_state.sample_images
        for col, img in zip(cols, col_imgs):
            col.image(img, use_column_width=True)


        # LOAD MODEL
        # ----------

        if not st.session_state.reached_test_phase:
            if app_method == "SPADE":
                model = SPADE(
                    k=3,
                    backbone_name=app_backbone,
                )
            elif app_method == "PaDiM":
                model = PaDiM(
                    d_reduced=75,
                    backbone_name=app_backbone,
                )
            elif app_method == "PatchCore":
                model = PatchCore(
                    f_coreset=.01, 
                    backbone_name=app_backbone,
                    coreset_eps=.95,
                )
            st.success(f"Loaded {app_method} model.")
        else:
            model = st.session_state.model
        
        # TRAINING
        # --------

        if not st.session_state.reached_test_phase:
            with st_stdout("info", "Setting up training ..."):            
                model.fit(DataLoader(train_dataset))

        # TESTING
        # -------

        if not st.session_state.reached_test_phase:
            st.session_state.reached_test_phase = True
            st.session_state.sample_images = col_imgs
            st.session_state.model = model
            st.session_state.train_dataset = train_dataset
            st.session_state.test_dataset = test_dataset
        
        st.session_state.test_idx = st.number_input(
            "Test sample index",
            min_value = 0,
            max_value = len(test_dataset)-1,
        )
        
        sample, *_ = test_dataset[st.session_state.test_idx]
        img_lvl_anom_score, pxl_lvl_anom_score = model.predict(sample.unsqueeze(0))
        score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
        if not manualRange:
            color_range = score_range
        show_pred(sample, img_lvl_anom_score, pxl_lvl_anom_score, color_range)
        st.write("pixel score min:{:.0f}".format(score_range[0]))
        st.write("pixel score max:{:.0f}".format(score_range[1]))

@contextmanager
def st_redirect(src, dst, msg):
    """https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602"""
    placeholder = st.info(msg)
    sleep(3)
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(b)
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write
            placeholder.empty()

@contextmanager
def st_stdout(dst, msg):
    """https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602"""
    with st_redirect(sys.stdout, dst, msg):
        yield

@contextmanager
def st_stderr(dst):
    """https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602"""
    with st_redirect(sys.stderr, dst):
        yield

if __name__ == "__main__":
    main()
