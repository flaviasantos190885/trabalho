import streamlit as st
from transformers import pipeline
import wikipedia
import warnings

# Ignorar warnings de modelos Hugging Face para uma saída mais limpa
warnings.filterwarnings("ignore")

# --- Configurações ---
# Modelo T5 para geração em PT (Fallback se Wikipedia falhar)
MODEL_FLAN = "google/flan-t5-base"

# Definir a linguagem da Wikipedia para Português
wikipedia.set_lang("pt")

# --- Funções de Inicialização (Cache) ---

# Usar st.cache_resource para carregar o pipeline apenas uma vez, 
# otimizando a performance do Streamlit.
@st.cache_resource
def load_generator_pipeline(model_name):
    """Carrega o pipeline de geração de texto (LLM)."""
    try:
        # Tenta detectar CUDA/GPU, se disponível
        device = 0 if st.session_state.get('use_cuda', False) else -1
        st.info(f"Carregando modelo {model_name}. Dispositivo: {'GPU' if device == 0 else 'CPU'}...")
        
        # O modelo Flan-T5 é text2text-generation, ideal para este tipo de tarefa
        gen_pipe = pipeline(
            "text2text-generation", 
            model=model_name, 
            tokenizer=model_name, 
            device=device
        )
        st.success("Modelo carregado com sucesso!")
        return gen_pipe
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de geração: {e}")
        return None

# Função de geração segura (Cópia da lógica original)
def safe_generate_text(pipe, prompt, max_new_tokens=150, deterministic=False):
    """Geração segura usando um pipeline Hugging Face."""
    gen_kwargs = {
        "max_new_tokens": max_new_tokens, 
        "early_stopping": True, 
        "no_repeat_ngram_size": 3, 
        "repetition_penalty": 1.2
    }
    if not deterministic:
        gen_kwargs.update({"do_sample": True, "top_p": 0.92, "temperature": 0.9})

    res = pipe(prompt, **gen_kwargs, num_return_sequences=1)
    
    if isinstance(res, list) and len(res) > 0:
        return res[0].get("generated_text") or res[0].get("text") or res[0].get("summary_text") or str(res[0])
    return ""


# --- Lógica Principal de Busca e Geração ---

def execute_text_generation(tema: str, gen_pipe, chosen_model: str = MODEL_FLAN):
    """
    Tenta buscar na Wikipedia (PT) e usa LLM como fallback.
    Retorna o texto gerado ou o resumo.
    """
    if not tema:
        st.warning("Por favor, digite um tema para buscar.")
        return ""

    st.subheader(f"Resultado para: **{tema}**")
    st.markdown("---")

    # 1. Tentar Wikipedia
    st.info(f"🔎 Buscando informações sobre '{tema}' na Wikipedia (PT)...")
    try:
        results = wikipedia.search(tema, results=3)
        
        if results:
            page = wikipedia.page(results[0])
            summary = page.summary
            
            # Limita aos 5 primeiros parágrafos
            paragraphs = summary.split("\n")
            resumo_final = "\n\n".join(paragraphs[:5]).strip()
            
            st.success("🟢 Resultado encontrado na **Wikipedia**:")
            st.code(resumo_final, language="text")
            return resumo_final
            
        else:
            st.warning("❌ Nenhum resultado encontrado na Wikipedia. Iniciando fallback (Geração LLM)...")
            
            
    except wikipedia.exceptions.PageError:
        st.warning(f"Erro: Página da Wikipedia não encontrada para '{tema}'. Iniciando fallback (Geração LLM)...")
        
    except Exception as e:
        st.error(f"Erro durante a busca na Wikipedia ({type(e).__name__}). Iniciando fallback (Geração LLM)...")


    # 2. Fallback - Geração automática (LLM)
    if gen_pipe:
        st.info("🧠 Gerando texto automaticamente (LLM)...")
        try:
            prompt = (
                f"Escreva um texto informativo e coerente em português sobre o artista ou tema musical '{tema}'. "
                "Use linguagem natural e ao menos 5 frases completas."
            )
            
            # Usando st.spinner para mostrar que está trabalhando
            with st.spinner('Processando...'):
                texto = safe_generate_text(gen_pipe, prompt, max_new_tokens=220, deterministic=False)
            
            if texto:
                st.success("🟢 Resultado gerado pelo modelo:")
                st.code(texto.strip(), language="text")
                return texto.strip()
            else:
                st.error("❌ Fallback de geração falhou ou retornou texto vazio.")
                return ""
                
        except Exception as e:
            st.error(f"(Erro) Fallback de geração de texto falhou: {e}")
            return ""
    else:
        st.error("❌ O pipeline do modelo de geração não foi carregado.")
        return ""

# --- Interface Streamlit ---
def main():
    st.set_page_config(page_title="Gerador de Conteúdo Musical NLP", layout="centered")
    
    st.title("🎶 Gerador de Conteúdo Musical (Wikipedia + LLM)")
    st.markdown("---")
    
    st.sidebar.header("Configurações")
    use_cuda = st.sidebar.checkbox("Usar GPU (CUDA)", value=torch.cuda.is_available() if 'torch' in locals() else False)
    st.session_state['use_cuda'] = use_cuda
    
    # Modelo pode ser alterado, mas o padrão é o recomendado
    chosen_model = st.sidebar.text_input("Modelo LLM (opcional):", value=MODEL_FLAN)
    
    # Carregar o pipeline (otimizado com cache)
    generator_pipeline = load_generator_pipeline(chosen_model)

    st.header("Entrada")
    tema = st.text_area(
        "Digite o nome de um artista, banda ou música:",
        placeholder="Ex: Marisa Monte", 
        height=100
    ).strip()

    st.markdown("---")

    if st.button("🚀 Buscar/Gerar Texto"):
        if generator_pipeline:
            # Envelopa a execução em um expander para organizar a saída
            with st.expander("Resultado da Análise", expanded=True):
                execute_text_generation(tema, generator_pipeline, chosen_model)
        else:
            st.warning("O modelo não está carregado. Verifique as configurações e tente novamente.")

if __name__ == "__main__":
    # Tenta importar torch para checar CUDA (ajuste conforme seu ambiente Streamlit)
    try:
        import torch
    except ImportError:
        st.warning("Aviso: O módulo 'torch' não foi encontrado. A detecção de GPU pode não funcionar.")
        
    main()