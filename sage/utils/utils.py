import os


def read_txt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
}

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")


WEB_SEARCH = {
    "gcp_bucket": os.getenv("GCP_BUCKET"),
    "serper_api_key": os.getenv("SERPER_API_KEY"),
}

CONTEXT_VLM_PROMPT = read_txt("sage/prompts/sampled_frames/context_vlm.txt")
SAMPLED_BASELINE_PROMPT = read_txt("sage/prompts/sampled_frames/baseline.txt")
SAMPLED_BASELINE_MINERVA_PROMPT = read_txt("sage/prompts/sampled_frames/baseline_minerva.txt")

SAGE_CONTEXT_VLM_PROMPT = read_txt("sage/prompts/sage/context_vlm.txt")
SAGE_ITERATIVE_REASONER_PROMPT = read_txt("sage/prompts/sage/iterative_reasoner.txt")
SAGE_ITERATIVE_REASONER_MSG_PROMPT = read_txt("sage/prompts/sage/iterative_reasoner_msg.txt")

ITERATIVE_REASONER_PROMPT = read_txt("sage/prompts/sampled_frames/iterative_reasoner.txt")
ITERATIVE_REASONER_MSG = read_txt("sage/prompts/sampled_frames/iterative_reasoner_msg.txt")
GENERATE_QUESTION_PROMPT = read_txt("sage/prompts/ques_gen.txt")

VISUAL_TEMPORAL_GROUNDING_PROMPT = read_txt("sage/prompts/visual_temporal_grounding.txt")
