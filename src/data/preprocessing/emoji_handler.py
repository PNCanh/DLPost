import re


def replace_emoji(
    text
):

    emoji_map = {

        "💰": " tien ",
        "🎁": " qua_tang ",
        "📢": " thong_bao ",
        "🔥": " hot ",
        "⚠️": " canh_bao ",
        "✅": " xac_nhan ",
        "❌": " tu_choi "
    }

    for k, v in emoji_map.items():

        text = text.replace(
            k,
            v
        )

    return text