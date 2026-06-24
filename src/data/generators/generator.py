import json
import random

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from configs import paths


class PostGenerator:

    SKIP_GENERATION_LABELS = [

        "fake_image"
    ]

    ENGLISH_PHRASES = [

        "work from home",

        "part time job",

        "limited offer",

        "earn money online",

        "investment opportunity",

        "special promotion",

        "remote working",

        "apply now"
    ]

    EMOJIS = [

        "🔥",

        "🚀",

        "🎉",

        "💰",

        "📢",

        "⚡",

        "💸",

        "✨"
    ]

    def __init__(self):

        self.template_root = paths.TEMPLATES_DIR

        self.output_root = paths.RAW_DIR

        self.labels = self.load_json(
            paths.LABELS_FILE
        )

        self.keywords = self.load_json(
            paths.SCAM_KEYWORDS_FILE
        )

        self.teencode = self.load_json(
            paths.TEENCODE_FILE
        )

        self.abbreviations = self.load_json(
            paths.ABBREVIATIONS_FILE
        )

        self.output_root.mkdir(

            parents=True,

            exist_ok=True
        )

    # ==================================================
    # LOADERS
    # ==================================================

    def load_json(
        self,
        file_path
    ):

        with open(
            file_path,
            encoding="utf8"
        ) as f:

            return json.load(f)

    # ==================================================
    # VALIDATION
    # ==================================================

    def validate_label(
        self,
        label_name
    ):

        valid_labels = (

            self.labels
            .get(
                "multiclass",
                {}
            )
            .keys()
        )

        if label_name not in valid_labels:

            raise ValueError(

                f"Invalid label: "
                f"{label_name}"
            )

    # ==================================================
    # RANDOM DATA
    # ==================================================

    def random_phone(self):

        return (

            "09"
            + "".join(

                random.choice(
                    "0123456789"
                )

                for _ in range(8)
            )
        )

    def random_salary(self):

        return random.choice([

            "8 triệu",

            "10 triệu",

            "15 triệu",

            "20 triệu",

            "30 triệu",

            "50 triệu"
        ])

    def random_url(self):

        return random.choice([

            "https://forms.gle/demo",

            "https://bit.ly/apply",

            "https://tinyurl.com/register",

            "https://zalo.me/group"
        ])

    # ==================================================
    # TEMPLATE
    # ==================================================

    def get_templates(
        self,
        label_name
    ):

        template_dir = (

            self.template_root /
            label_name
        )

        if not template_dir.exists():

            return []

        return list(

            template_dir.glob(
                "*.txt"
            )
        )

    def random_template(
        self,
        label_name
    ):

        templates = (

            self.get_templates(
                label_name
            )
        )

        if not templates:

            return None

        template = random.choice(
            templates
        )

        return template.read_text(
            encoding="utf8"
        )

    # ==================================================
    # KEYWORDS
    # ==================================================

    def get_flattened_keywords(
        self,
        label_name
    ):
        label_data = self.keywords.get(label_name, {})
        if not label_data:
            return []
            
        if isinstance(label_data, dict):
            flattened = []
            if "without_value" in label_data:
                flattened.extend(label_data["without_value"])
            if "with_value" in label_data:
                for key, values in label_data["with_value"].items():
                    if values:
                        flattened.append(f"{key} {random.choice(values)}")
                    else:
                        flattened.append(key)
            return flattened
        elif isinstance(label_data, list):
            return label_data
        return []

    def random_keyword(
        self,
        label_name
    ):

        keywords = self.get_flattened_keywords(label_name)

        if not keywords:

            return ""

        return random.choice(
            keywords
        )

    def fallback_text(
        self,
        label_name
    ):

        keywords = self.get_flattened_keywords(label_name)

        if not keywords:

            return label_name

        samples = random.sample(

            keywords,

            min(
                3,
                len(keywords)
            )
        )

        return " ".join(
            samples
        )

    # ==================================================
    # TEMPLATE RENDER
    # ==================================================

    def render_template(
        self,
        template,
        label_name
    ):

        return template.format(

            keyword=
                self.random_keyword(
                    label_name
                ),

            phone=
                self.random_phone(),

            salary=
                self.random_salary(),

            url=
                self.random_url()
        )

    # ==================================================
    # AUGMENT TEXT
    # ==================================================

    def inject_teencode(
        self,
        text
    ):

        items = list(
            self.teencode.items()
        )

        random.shuffle(items)

        for normal, teen in items[:5]:

            if random.random() < 0.3:

                text = text.replace(
                    normal,
                    teen
                )

        return text

    def inject_abbreviations(
        self,
        text
    ):

        items = list(
            self.abbreviations.items()
        )

        random.shuffle(items)

        for full, short in items[:5]:

            if random.random() < 0.3:

                text = text.replace(
                    full,
                    short
                )

        return text

    def inject_english(
        self,
        text
    ):

        if random.random() < 0.2:

            text += (

                "\n" +

                random.choice(
                    self.ENGLISH_PHRASES
                )
            )

        return text

    def inject_emoji(
        self,
        text
    ):

        if random.random() < 0.5:

            text = (

                random.choice(
                    self.EMOJIS
                )

                + " "

                + text
            )

        return text

    def inject_no_accent(
        self,
        text
    ):

        # if random.random() < 0.15:

        #     return unidecode(text)

        return text

    # ==================================================
    # BUILD TEXT
    # ==================================================

    def build_text(
        self,
        label_name
    ):

        template = (

            self.random_template(
                label_name
            )
        )

        if template:

            text = self.render_template(

                template,

                label_name
            )

        else:

            text = self.fallback_text(
                label_name
            )

        text = self.inject_teencode(
            text
        )

        text = self.inject_abbreviations(
            text
        )

        text = self.inject_english(
            text
        )

        text = self.inject_emoji(
            text
        )

        text = self.inject_no_accent(
            text
        )

        return text

    # ==================================================
    # INDEX
    # ==================================================

    def next_post_index(self):

        folders = list(

            self.output_root.glob(
                "post*"
            )
        )

        if not folders:

            return 1

        indexes = []

        for folder in folders:

            digits = "".join(

                c

                for c in folder.name

                if c.isdigit()
            )

            if digits:

                indexes.append(
                    int(digits)
                )

        return max(indexes) + 1

    # ==================================================
    # CREATE POST
    # ==================================================

    def create_post(
        self,
        label_name,
        index
    ):

        post_id = (

            f"post_{index:05d}"
        )

        post_dir = (

            self.output_root /
            f"post{index:05d}"
        )

        post_dir.mkdir(

            parents=True,

            exist_ok=True
        )

        text = self.build_text(
            label_name
        )

        binary_label = (

            0

            if label_name
            == "legitimate"

            else 1
        )

        with open(

            post_dir /
            "text.txt",

            "w",

            encoding="utf8"
        ) as f:

            f.write(text)

        payload = {

            "post_id":
                post_id,

            "text":
                "text.txt",

            "binary_label":
                binary_label,

            "multi_label":
                label_name,
        }

        with open(

            post_dir /
            "post.json",

            "w",

            encoding="utf8"
        ) as f:

            json.dump(

                payload,

                f,

                ensure_ascii=False,

                indent=4
            )

    # ==================================================
    # GENERATE
    # ==================================================

    def generate(
        self,
        label_name,
        quantity
    ):

        self.validate_label(
            label_name
        )

        if (

            label_name

            in

            self.SKIP_GENERATION_LABELS
        ):

            print(
                f"Skip {label_name}"
            )

            return

        start_index = (
            self.next_post_index()
        )

        for i in range(quantity):

            self.create_post(

                label_name,

                start_index + i
            )

        print(

            f"Generated "

            f"{quantity} "

            f"{label_name}"
        )

    def generate_multiple(
        self,
        config
    ):

        for (

            label_name,

            quantity

        ) in config.items():

            self.generate(

                label_name,

                quantity
            )


if __name__ == "__main__":

    generator = PostGenerator()

    generator.generate_multiple({

        "job_scam": 20 ,

        "investment_scam": 20,

        "sale_scam": 20,

        "prize_scam": 20,

        "fake_course": 20,

        "suspicious": 20,

        "legitimate": 20
    })