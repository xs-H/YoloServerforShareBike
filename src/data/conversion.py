from pathlib import Path  # 路径操作工具
import yaml  # Yaml配置文件处理
import shutil  # 文件操作工具
import logging  # 日志记录
import xml.etree.ElementTree as ET  # XML解析库
from sklearn.model_selection import train_test_split  # 数据集分割, pip install scikit-learn

# 配置日志记录级别和格式
logging.basicConfig(level=logging.INFO)
# 创建一个日志记录器
logger = logging.getLogger(__name__)


class XML2YOLOConverter:
    """XML标准转YOLO格式的主类"""

    def __init__(self, raw_data_path, output_path, classes, train_ratio=0.8, val_ratio=0.1):
        """
        初始化转换器
        :param raw_data_path: 原始数据根目录路径
        :param output_path: 输出根目录路径
        :param classes: 类别名称列表
        :param train_ratio: 训练集比例（默认0.8）
        :param val_ratio: 验证集比例（默认0.1）
        """
        self.raw_data_path = Path(raw_data_path)

        # 设置输出路径
        self.output_path = Path(output_path) / "data"
        # 设置配置文件路径
        self.config_path = Path(output_path) / "configs"
        self.classes = classes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        # 计算测试集的比例
        self.test_ratio = 1.0 - train_ratio - val_ratio
        # 构建原始图像和标注路径
        self.raw_images_path = self.raw_data_path / "raw" / "images"
        self.raw_annotations_path = self.raw_data_path / "raw" / "annotations"

        # 检查标注目录和图像目录是否存在
        if not self.raw_annotations_path.exists():
            logger.error(f"标注目录{self.raw_annotations_path}不存在")
            raise FileNotFoundError(f"标注目录{self.raw_annotations_path}不存在")
        if not self.raw_images_path.exists():
            logger.error(f"图像目录{self.raw_images_path}不存在")
            raise FileNotFoundError(f"图像目录{self.raw_images_path}不存在")

        # 初始化输出路径字典
        # 训练集、验证集、测试集的图像路径
        self.output_images = {
            "train": self.output_path / "train" / "images",
            "val": self.output_path / "val" / "images",
            "test": self.output_path / "test" / "images"
        }

        # 训练集、验证集、测试集的标签路径
        self.output_labels = {
            "train": self.output_path / "train" / "labels",
            "val": self.output_path / "val" / "labels",
            "test": self.output_path / "test" / "labels"
        }

        # 创建所有输出目录
        for split in ["train", "val", "test"]:
            #  创建图像目录
            self.output_images[split].mkdir(parents=True, exist_ok=True)
            self.output_labels[split].mkdir(parents=True, exist_ok=True)
        # 创建配置文件目录
        self.config_path.mkdir(parents=True, exist_ok=True)

    def convert(self):
        """执行转换功能的主要方法"""
        # 获取所有的xml标注文件(支持.xml和.XML扩展名)
        xml_files = list(self.raw_annotations_path.glob("*.xml")) + \
                    list(self.raw_annotations_path.glob("*.XML"))

        # 检查是否查找到XML文件
        if not xml_files:
            logger.error("未找到任何标注XML文件")
            return
        # 准备手机有效的数据
        image_paths = []  # 存储有效的图像路径
        valid_xml_files = []  # 存储有效的XML文件路径
        # 支持的图像扩展名
        img_extension = [".jpg", ".jpeg", ".png"]

        # 遍历所有的XML文件，查找对应的图像
        for xml_file in xml_files:
            for ext in img_extension:
                # 构建可能都图像文件名（与XML同名的不同扩展名）
                img_name = xml_file.stem + ext
                img_path = self.raw_images_path / img_name
                # 如果图像存在
                if img_path.exists():
                    image_paths.append(img_path)
                    valid_xml_files.append(xml_file)
                    break
            else:
                logger.warning(f"图像{xml_file.stem}不存在，跳过")
        # 检查是否有有效的图像-xml对
        if not valid_xml_files:
            logger.error("没有找到任何有效的xml 和 图像 对")
            return
        # 分割数据集为训练集，验证集，测试集
        train_xmls, temp_xmls, train_imgs, temp_imgs = train_test_split(
            valid_xml_files, image_paths, train_size=self.train_ratio, random_state=42)
        # 调整验证几比例，相对于剩余的部分
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_xmls, test_xmls,val_imgs,test_imgs = train_test_split(
            temp_xmls, temp_imgs, train_size=val_ratio_adjusted, random_state= 42,
        )
        # 处理每个数据集的分割
        self._process_split(train_xmls, train_imgs, "train")
        self._process_split(val_xmls, val_imgs, "val")
        self._process_split(test_xmls, test_imgs, "test")

        # 准备Yolo配置文件的内容
        data_yaml = {
            "train": str(self.output_images["train"]),
            "val": str(self.output_images["val"]),
            "test": str(self.output_images["test"]),
            "nc": len(self.classes),
            "names": self.classes
        }
        # data_yaml = {
        #     "train": "../data/train/images",
        #     "val": "../data/val/images",
        #     "test": "../data/test/images",
        #     "nc": len(self.classes),
        #     "names": self.classes
        # }
        # 配置文件的路径
        yaml_path = self.config_path / "data.yaml"
        # 写入YAML文件
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
        # 记录完成信息
        logger.info(f"数据集转换完成")
        logger.info(f"data_yaml已经生存位置在：{yaml_path}")
        logger.info(f"数据集已经保存在：{self.output_path}")
        logger.info(f"   train: {data_yaml['train']}")
        logger.info(f"   val: {data_yaml['val']}")
        logger.info(f"   test: {data_yaml['test']}")
        logger.info(f"   nc: {data_yaml['nc']}")
        logger.info(f"   names: {data_yaml['names']}")



    # 处理每个数据集的分割
    def _process_split(self, xml_files, img_paths, split):
        """
        处理单个数据集分割（训练，验证，测试）
        :param self:
        :param xml_files: 改分割单xml文件列表
        :param img_paths: 对应的图像路径列表
        :param split: 分割单名称，train,test,val
        :return:
        """
        # 复制图像到目标目录
        for img_path in img_paths:
            # 目标图像路径
            new_img_path = self.output_images[split] / img_path.name
            # 复制图像文件
            shutil.copy(img_path, new_img_path)
        for xml_file in xml_files:
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # 获取图像的尺寸信息
            size = root.find("size")
            if size is None:
                logger.error(f"XML文件{xml_file}缺少<size>标签")
                continue
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

            # 准备Yolo格式标签了
            yolo_labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in self.classes:
                    logger.warning(f"类别{name}不在定义列表当中")
                    continue
                class_id = self.classes.index(name)
                # 获取边界信息
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    logger.warning(f"对象{name}在{xml_file}中缺少<bndbox>标签，跳过")
                    continue
                # 提取边界框坐标
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                # 转换为Yolo格式
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # 构建Yolo格式标签行
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            # 构建标签的文件路径
            label_file = self.output_labels[split] / (xml_file.stem + ".txt")
            # 写入标签文件
            with open(label_file, "w") as f:
                if yolo_labels:
                    f.write("\n".join(yolo_labels))
                else:
                    f.write("")
            # 记录日志处理
            logger.debug(f"处理{xml_file} -> {split}分割")


if __name__ == "__main__":
    # 定义类别列表
    classes = ["safety_helmet", "reflective_vest", "person", "head", "ordinary_clothes"]
    # 获取基础路径
    base_path = Path(__file__).parent.parent.parent
    # 设置原始数据路径
    raw_data_path = base_path / "data"

    # 设置输出路径
    output_path = base_path
    # 创建转换器对象
    converter = XML2YOLOConverter(
        raw_data_path= raw_data_path,
        output_path = output_path,
        classes = classes,
        train_ratio= 0.8,
        val_ratio= 0.1,
    )
    # 执行转换
    converter.convert()