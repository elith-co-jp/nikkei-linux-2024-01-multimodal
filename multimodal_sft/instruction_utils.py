from langchain.prompts import PromptTemplate


def fill_template(template: str, info_dict: dict) -> str:
    """インストラクション作成用のテンプレートを埋める関数

    Args:
        template (str): テンプレートテキスト
        info_dict (dict): テンプレートに埋める情報

    Returns:
        str: 埋めたテキスト
    """
    prompt = PromptTemplate.from_template(template)
    keys = prompt.input_variables
    input_dict = {key: info_dict[key] for key in keys}
    return prompt.format_prompt(**input_dict).text
