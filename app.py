import streamlit as st
import advertools as adv
import pandas as pd
import numpy as np
import tempfile
import json
import os
import concurrent.futures
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from urllib.parse import urlparse

# --- Configuration & Caching ---
st.set_page_config(page_title="Semantic Link Automator", layout="wide")
st.title("Automated Semantic Internal Linking Tool")

@st.cache_resource
def load_model():
    """Loads the sentence transformer model once and caches it in memory."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_model()

# --- UI Inputs ---
with st.sidebar:
    st.header("Settings")
    gemini_api_key = st.text_input("Gemini API Key", type="password")

    st.subheader("Site Discovery")
    sitemap_url = st.text_input("Sitemap URL", placeholder="https://example.com/sitemap.xml")

    st.subheader("Target Pages (Incoming Links)")
    target_urls_input = st.text_area("URLs to build links TO (one per line)", height=150)

    allow_new_copy = st.checkbox("Allow suggestions with NEW copy (1 line)")

    run_button = st.button("Generate Link Suggestions")

    st.markdown("---")
    st.subheader("Cache Management")
    clear_cache_button = st.button("Clear Cache for Domain")

# --- Helper Functions ---
def parse_urls(text_input):
    return [url.strip() for url in text_input.split('\n') if url.strip()]

def get_urls_from_sitemap(sitemap):
    try:
        sitemap_df = adv.sitemap_to_df(sitemap)
        return sitemap_df['loc'].dropna().tolist()
    except Exception as e:
        st.error(f"Failed to parse sitemap: {e}")
        return []

def crawl_and_filter(urls):
    with tempfile.NamedTemporaryFile(suffix='.jl', delete=False) as tmp_file:
        filepath = tmp_file.name

    try:
        adv.crawl(urls, filepath, custom_settings={'LOGLEVEL': 'ERROR'})
        df = pd.read_json(filepath, lines=True)

        # Filter for indexable, 200 status pages
        if 'status' in df.columns:
            df = df[df['status'] == 200]

        if 'meta_robots' in df.columns:
            df = df[~df['meta_robots'].str.contains('noindex', na=False, case=False)]

        if 'canonical' in df.columns:
            df = df[df['canonical'].isna() | (df['canonical'] == df['url'])]

        if 'body_text' not in df.columns:
            df['body_text'] = df.get('title', '')

        df['body_text'] = df['body_text'].fillna('')

        # Ensure we capture the outgoing links for our new filter
        if 'links_url' not in df.columns:
            df['links_url'] = ''
        else:
            df['links_url'] = df['links_url'].fillna('')

        return df[['url', 'title', 'body_text', 'links_url']]
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def chunk_text(text, chunk_size=150):
    words = str(text).split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [c for c in chunks if len(c.split()) > 20]

def get_gemini_suggestions(target_url, source_url, source_chunk, allow_new_copy):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')

    prompt = f"""
    You are an expert SEO content strategist. Find a natural way to add an internal link from the "Source Paragraph" to the "Target Page".

    Target Page URL: {target_url}
    Source Page URL: {source_url}

    Source Paragraph (Context):
    {source_chunk}

    INSTRUCTIONS:
    1. The anchor text MUST be highly relevant to the Target Page.
    2. Suggest a link using EXISTING copy from the source paragraph. Provide the exact existing sentence and specify which words should be the anchor text.
    """

    if allow_new_copy:
        prompt += "\n3. Suggest a link using NEW copy. Write exactly ONE new sentence that fits naturally into the paragraph context, and specify the anchor text."

    prompt += """
    Output the response in clean JSON format:
    - "existing_copy_sentence": The full sentence from the text.
    - "existing_copy_anchor": The exact words to hyperlink.
    """
    if allow_new_copy:
        prompt += """
    - "new_copy_sentence": The newly written one-line sentence.
    - "new_copy_anchor": The exact words to hyperlink.
    """

    prompt += "\nReturn ONLY valid JSON. Do not use Markdown blocks."

    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"error": str(e)}

# --- Cache Management Execution ---
if clear_cache_button and sitemap_url:
    domain_name = urlparse(sitemap_url).netloc.replace("www.", "")
    cache_dir = "cache"

    paths_to_remove = [
        os.path.join(cache_dir, f"{domain_name}_faiss.index"),
        os.path.join(cache_dir, f"{domain_name}_chunks.pkl"),
        os.path.join(cache_dir, f"{domain_name}_url_text.pkl"),
        os.path.join(cache_dir, f"{domain_name}_url_links.pkl") # NEW: clear the links cache too
    ]

    cleared = False
    for path in paths_to_remove:
        if os.path.exists(path):
            os.remove(path)
            cleared = True

    if cleared:
        st.sidebar.success(f"Cache cleared for {domain_name}!")
    else:
        st.sidebar.info("No cache found to clear.")

# --- Main Workflow ---
if run_button:
    if not gemini_api_key:
        st.error("Please provide a Gemini API Key.")
        st.stop()

    target_urls = parse_urls(target_urls_input)

    if not sitemap_url or not target_urls:
        st.error("Please provide a Sitemap URL and Target URLs.")
        st.stop()

    domain_name = urlparse(sitemap_url).netloc.replace("www.", "")
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    faiss_path = os.path.join(cache_dir, f"{domain_name}_faiss.index")
    chunk_data_path = os.path.join(cache_dir, f"{domain_name}_chunks.pkl")
    url_text_path = os.path.join(cache_dir, f"{domain_name}_url_text.pkl")
    url_links_path = os.path.join(cache_dir, f"{domain_name}_url_links.pkl")

    # --- Step 1 & 2: Load Cache OR Crawl & Save ---
    if os.path.exists(faiss_path) and os.path.exists(chunk_data_path) and os.path.exists(url_text_path) and os.path.exists(url_links_path):
        st.info(f"Loading cached data for {domain_name} from local disk...")

        faiss_index = faiss.read_index(faiss_path)
        with open(chunk_data_path, 'rb') as f:
            chunk_data = pickle.load(f)
        with open(url_text_path, 'rb') as f:
            url_to_text = pickle.load(f)
        with open(url_links_path, 'rb') as f:
            url_to_links = pickle.load(f)

        valid_targets = [u for u in target_urls if u in url_to_text and url_to_text[u]]
        if not valid_targets:
            st.error("Target URLs not found in the cached data. Please clear the cache and re-crawl.")
            st.stop()

    else:
        with st.spinner(f"No cache found. Crawling {domain_name} (this may take a moment)..."):
            all_site_urls = get_urls_from_sitemap(sitemap_url)
            if not all_site_urls:
                st.stop()

            urls_to_crawl = list(set(all_site_urls + target_urls))
            filtered_df = crawl_and_filter(urls_to_crawl)

            url_to_text = dict(zip(filtered_df['url'], filtered_df['body_text']))
            url_to_links = dict(zip(filtered_df['url'], filtered_df['links_url']))

            valid_targets = [u for u in target_urls if u in url_to_text and url_to_text[u]]
            if not valid_targets:
                st.error("Could not extract content from the target URLs. Ensure they are accessible.")
                st.stop()

        with st.spinner("Generating semantic embeddings and saving to local cache..."):
            chunk_data = []
            source_df = filtered_df[filtered_df['url'].isin(all_site_urls)]

            for _, row in source_df.iterrows():
                chunks = chunk_text(row['body_text'])
                for chunk in chunks:
                    chunk_data.append({
                        'url': row['url'],
                        'text': chunk
                    })

            chunk_texts = [c['text'] for c in chunk_data]
            chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
            chunk_embeddings = np.array(chunk_embeddings).astype('float32')

            dimension = chunk_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(chunk_embeddings)

            faiss.write_index(faiss_index, faiss_path)
            with open(chunk_data_path, 'wb') as f:
                pickle.dump(chunk_data, f)
            with open(url_text_path, 'wb') as f:
                pickle.dump(url_to_text, f)
            with open(url_links_path, 'wb') as f:
                pickle.dump(url_to_links, f)

            st.success(f"Successfully cached data for {domain_name}!")

    # --- Step 3: Semantic Search ---
    st.info("Indexing Complete! Finding semantic matches and filtering out existing links...")

    tasks = []

    for target_url in valid_targets:
        target_text = url_to_text[target_url]
        target_summary = ' '.join(target_text.split()[:300])

        target_vector = embedding_model.encode([target_summary]).astype('float32')

        # INCREASED from 15 to 50: We need a wider net in case the top matches already link to the target
        distances, indices = faiss_index.search(target_vector, 50)

        found_sources = set()
        found_matches = []

        # Clean target URL for exact matching
        target_parsed = urlparse(target_url)
        target_url_clean = target_url.strip().strip('/')
        target_domain = target_parsed.netloc.replace("www.", "")
        target_path = target_parsed.path.strip('/')

        for idx, dist in zip(indices[0], distances[0]):
            match = chunk_data[idx]
            match_url = match['url']

            # Skip if it's the target page itself, or if we already picked a chunk from this URL
            if match_url == target_url or match_url in found_sources:
                continue

            # --- NEW: Check if the source already links to the target ---
            source_links_str = url_to_links.get(match_url, '')
            source_links = [link.strip().strip('/') for link in source_links_str.split('@@') if link]

            link_exists = False
            for link in source_links:
                link_clean = link

                # Check 1: Exact string match
                if target_url_clean == link_clean:
                    link_exists = True
                    break

                # Check 2: Relative URL or structural match (e.g. trailing slash differences)
                link_parsed = urlparse(link)
                link_domain = link_parsed.netloc.replace("www.", "")
                link_path = link_parsed.path.strip('/')

                if (not link_domain or link_domain == target_domain) and (link_path == target_path):
                    link_exists = True
                    break

            if link_exists:
                continue # Skip to the next closest semantic match!
            # -------------------------------------------------------------

            found_sources.add(match_url)
            found_matches.append((match_url, match['text'], dist))

            if len(found_matches) >= 3:
                break

        for match_url, chunk_text_match, dist in found_matches:
            tasks.append({
                'target_url': target_url,
                'source_url': match_url,
                'source_chunk': chunk_text_match,
                'distance': dist
            })

    # --- Step 4: Execute Tasks Concurrently ---
    results_map = {}
    progress_bar = st.progress(0)

    if not tasks:
        st.warning("Could not find any pages that don't already link to your target(s)!")
        st.stop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {
            executor.submit(
                get_gemini_suggestions,
                task['target_url'],
                task['source_url'],
                task['source_chunk'],
                allow_new_copy
            ): task for task in tasks
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            progress_bar.progress(completed / len(tasks))

            try:
                suggestion = future.result()
            except Exception as e:
                suggestion = {"error": str(e)}

            if task['target_url'] not in results_map:
                results_map[task['target_url']] = []

            results_map[task['target_url']].append({
                'source_url': task['source_url'],
                'distance': task['distance'],
                'suggestion': suggestion
            })

    # --- Step 5: Render UI and Export CSV ---
    csv_data = []

    for target_url in valid_targets:
        if target_url not in results_map:
            continue

        st.markdown("---")
        st.header(f"Target Page: `{target_url}`")

        mapped_sources = sorted(results_map[target_url], key=lambda x: x['distance'])

        for item in mapped_sources:
            source_url = item['source_url']
            suggestion = item['suggestion']
            formatted_suggestion = ""

            with st.expander(f"Source Page: {source_url}", expanded=True):
                if "error" in suggestion:
                    st.error(f"Error fetching from Gemini: {suggestion['error']}")
                    formatted_suggestion = f"Error: {suggestion['error']}"
                else:
                    st.subheader("Option 1: Using Existing Copy")
                    existing_sentence = suggestion.get('existing_copy_sentence', 'N/A')
                    existing_anchor = suggestion.get('existing_copy_anchor', 'N/A')
                    st.write(f"**Sentence:** {existing_sentence}")
                    st.write(f"**Anchor Text:** `{existing_anchor}`")

                    formatted_suggestion += f"EXISTING COPY\nSentence: {existing_sentence}\nAnchor: {existing_anchor}"

                    if allow_new_copy and "new_copy_sentence" in suggestion:
                        st.subheader("Option 2: Using New Copy")
                        new_sentence = suggestion.get('new_copy_sentence', 'N/A')
                        new_anchor = suggestion.get('new_copy_anchor', 'N/A')
                        st.write(f"**Suggested New Line:** {new_sentence}")
                        st.write(f"**Anchor Text:** `{new_anchor}`")

                        formatted_suggestion += f"\n\nNEW COPY\nSentence: {new_sentence}\nAnchor: {new_anchor}"

            csv_data.append({
                "Origin page": source_url,
                "Destination page": target_url,
                "suggested link": formatted_suggestion
            })

    if csv_data:
        st.markdown("---")
        st.subheader("Export Results")
        export_df = pd.DataFrame(csv_data)

        st.download_button(
            label="Download CSV Export",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="semantic_linking_suggestions.csv",
            mime="text/csv"
        )
