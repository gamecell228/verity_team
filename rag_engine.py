import os
import gdown
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

REPORTS_DIR = "./data/alfa_reports"
CHROMA_DIR = "./chroma_db"
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/15xZTLF1lZ65p6o5xpNA_fRbxjoV-UlG1?usp=sharing"

embedding_model = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")

def download_reports_from_gdrive():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.listdir(REPORTS_DIR):
        print("[RAG] Скачиваем аналитические отчеты Альфа-Банка из Google Drive...")
        try:
            gdown.download_folder(url=GDRIVE_FOLDER_URL, output=REPORTS_DIR, quiet=False, use_cookies=False)
        except Exception as e:
            print(f"[RAG Warning] Не удалось скачать напрямую через gdown ({e}). Создаем демонстрационные отчеты...")
            create_fallback_reports()
    else:
        print("[RAG] Папка с отчетами уже существует.")

def create_fallback_reports():
    mock_reports = {
        "Ключевая_ставка_14.txt": "Центр макроэкономических исследований Альфа-Банка: Совет директоров Банка России сохранил ключевую ставку на уровне 14.0%. Риторика ужесточилась. Мы прогнозируем паузу в снижении ставок минимум до 1 полугодия 2027 года. Рекомендуем корпоративным клиентам фиксацию ставок и размещение остатков в овернайт-депозиты под 13.5%.",
        "Обзор_рынков_и_ВЭД.txt": "Альфа-Банк Research: Логистические издержки по азиатскому направлению (Китай) выросли на 8%. Наблюдаются задержки платежей через страны Прибалтики (Литва). Рекомендуется переход на прямые корреспондентские счета в юанях и альтернативные хабы расчетов.",
        "Макростатистика_7М26_рынок_труда.txt": "Альфа-Банк Research: Данные рынка труда не подтверждают охлаждения экономики. Безработица на историческом минимуме, сохраняется давление на фонд оплаты труда (ФОТ)."
    }
    for fname, text in mock_reports.items():
        with open(os.path.join(REPORTS_DIR, fname), "w", encoding="utf-8") as f:
            f.write(text)

def get_or_build_vectorstore() -> Chroma:

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("[RAG] Загружаем готовую векторную базу из диска...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

    download_reports_from_gdrive()
    print("[RAG] Индексация документов отчетов Альфа-Банка...")
    
    docs: List[Document] = []
    for root, _, files in os.walk(REPORTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    docs.extend(loader.load())
                elif file.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())
            except Exception as e:
                print(f"[RAG Warning] Ошибка чтения {file}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print(f"[RAG] Успешно проиндексировано {len(splits)} фрагментов макроаналитики.")
    return vectorstore

vector_db = get_or_build_vectorstore()

def query_alfa_research(query_text: str, k: int = 2) -> List[str]:

    try:
        results = vector_db.similarity_search(query_text, k=k)
        return [f"[Отчет: {r.metadata.get('source', 'ЦМИ Альфа-Банк')}]: {r.page_content}" for r in results]
    except Exception as e:
        return [f"Аналитика Альфа-Банка: консервативный макропрогноз. (Ошибка поиска: {e})"]