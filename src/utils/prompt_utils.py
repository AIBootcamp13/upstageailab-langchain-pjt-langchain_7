from pathlib import Path

from langchain_core.prompts import PromptTemplate
from jinja2 import Environment, FileSystemLoader


def load_prompt(file_path: Path, template_dir: Path = Path("prompts")) -> PromptTemplate:
    """
    Jinja 파일에서 프롬프트 템플릿을 로드합니다.
    Args:
        file_path: Jinja 파일 경로
        template_dir: 템플릿 디렉토리 (기본값: "prompts")
    Returns:
        PromptTemplate 객체
    """
    file_path = template_dir / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(file_path.name)
    
    # Render the template without variables to resolve inheritance
    rendered_template = template.render()

    # Create PromptTemplate from rendered string
    return PromptTemplate.from_template(rendered_template)