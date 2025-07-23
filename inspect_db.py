import os
import pickle
import json

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")

def inspect_database():
    """Loads and inspects the contents of the metadata store."""
    print(f"[*] Attempting to load metadata from '{METADATA_PATH}'...")
    
    try:
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        print(f"[+] Successfully loaded metadata store.")
    except FileNotFoundError:
        print(f"\n[!] FATAL: Metadata file not found. Please run 'process_data.py' first.")
        return
    except Exception as e:
        print(f"\n[!] FATAL: Could not load the metadata file. Error: {e}")
        return

    total_items = len(metadata_store)
    if total_items == 0:
        print("\n[!] The metadata store is EMPTY. No data has been loaded.")
        return
        
    print(f"\n[*] The store contains metadata for {total_items} total items (vectors).")
    print("[*] Retrieving the first 10 items for inspection...")

    print("\n" + "="*80)
    print("                  METADATA INSPECTION REPORT")
    print("="*80)

    for i, metadata in enumerate(metadata_store[:10]):
        print(f"\n--- RECORD {i+1} ---")
        if not metadata:
            print("  [!] This record has NO METADATA.")
            continue
            
        for key, value in metadata.items():
            if key == 'original_text' and isinstance(value, str) and len(value) > 200:
                print(f"  - {key:<20}: '{value[:200].replace(chr(10), ' ')}...'")
            else:
                print(f"  - {key:<20}: {value}")
                
    print("\n" + "="*80)

if __name__ == "__main__":
    inspect_database()