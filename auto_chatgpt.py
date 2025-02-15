import time
import os
import pyperclip
import pyautogui
import pygetwindow as gw
import re

# Nécessite `pip install pyperclip pyautogui pygetwindow`

EDIFACT_FILE_PATH = "C:/EDIFACT/test.edi"
PROMPT = ("Corrige ce fichier EDIFACT et renvoie uniquement la "
          "version corrigée sous forme brute, sans explication ni commentaire.")

def bring_chatgpt_to_front():
    """Tente de mettre la fenêtre ChatGPT au premier plan."""
    try:
        for window in gw.getWindowsWithTitle("ChatGPT"):
            window.activate()
            print("[INFO] Fenêtre ChatGPT activée.")
            return True
    except Exception as e:
        print(f"[ERREUR] Impossible de trouver ou d'activer la fenêtre ChatGPT: {e}")
    return False

def extract_edifact_data(full_response: str) -> str:
    """Extrait le contenu EDIFACT après 'ChatGPT a dit :' et s'arrête après 'UNZ+1+000000001''."""
    # 1) Retirer tout ce qui précède "ChatGPT a dit :"
    marker_chatgpt = "ChatGPT a dit :"
    index_chatgpt = full_response.find(marker_chatgpt)
    if index_chatgpt != -1:
        text = full_response[index_chatgpt + len(marker_chatgpt):].strip()
    else:
        text = full_response.strip()

    # 2) Retirer tout ce qui suit "UNZ+1+000000001'"
    end_marker = "UNZ+1+000000001'"
    index_end = text.find(end_marker)
    if index_end != -1:
        text = text[:index_end + len(end_marker)].strip()

    return text

def main():
    if not os.path.exists(EDIFACT_FILE_PATH):
        print(f"[ERREUR] Le fichier {EDIFACT_FILE_PATH} est introuvable.")
        return

    # Lecture du contenu du fichier EDIFACT existant
    with open(EDIFACT_FILE_PATH, 'r', encoding='utf-8') as f:
        file_content = f.read().strip()

    # Génération du prompt complet pour ChatGPT
    full_prompt = f"{PROMPT}\n\n{file_content}"

    # Copie du prompt dans le presse-papier
    pyperclip.copy(full_prompt)

    # Tente de placer la fenêtre ChatGPT au premier plan
    if not bring_chatgpt_to_front():
        print("[ERREUR] ChatGPT n'est pas ouvert ou inaccessible.")
        return

    # Pause pour laisser l'utilisateur voir la fenêtre s'activer
    time.sleep(3)

    # Coller le texte et envoyer la demande à ChatGPT
    pyautogui.hotkey("ctrl", "v")
    pyautogui.press("enter")

    print("[INFO] Message envoyé à ChatGPT, en attente de réponse...")
    # On attend un peu plus longtemps pour que ChatGPT ait le temps de générer la réponse
    time.sleep(20)

    # Simule un triple clic pour sélectionner le texte, puis Ctrl+A et Ctrl+C pour copier la réponse
    pyautogui.click(x=500, y=500, clicks=3, interval=0.2)
    time.sleep(1)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(1)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(1)

    # Récupération du contenu corrigé depuis le presse-papier
    corrected_content = pyperclip.paste()

    # Extraction de la partie EDIFACT en enlevant le bas
    edifact_data = extract_edifact_data(corrected_content)

    if edifact_data:
        # Écriture du fichier corrigé
        corrected_file_path = EDIFACT_FILE_PATH.replace(".edi", "_corrige.edi")
        with open(corrected_file_path, "w", encoding="utf-8") as f:
            f.write(edifact_data)
        print(f"[INFO] Fichier corrigé enregistré sous : {corrected_file_path}")
    else:
        print("[ERREUR] Aucun contenu EDIFACT détecté.")

if __name__ == '__main__':
    main()
