import sys
import os
import numpy as np

# إضافة مسار المشروع للنظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_engine import CogniPro

def fix_japan_knowledge():
    platform = CogniPro()
    print("   [v9.5 Rescue] Re-anchoring Japan Knowledge Shard...")
    
    # Grounded factual description for Japan
    japan_desc = (
        "Japan (Nippon/Nihon) is an island country in East Asia, located in the northwest Pacific Ocean. "
        "It is bordered by the Sea of Japan to the west and extends from the Sea of Okhotsk in the north "
        "to the East China Sea and Taiwan in the south. Japan is part of the Pacific Ring of Fire and "
        "comprises an archipelago of 6,852 islands, with Tokyo as its capital and largest city."
    )
    
    # Manually save knowledge, overwriting the JIT anchor
    vec = platform.get_sentence_vector(japan_desc)
    platform.long_term.save_knowledge(
        concept="Japan", 
        info=japan_desc, 
        vector=vec, 
        category="Linguistic", 
        label="Wikipedia: Geography",
        is_locked=True # Lock this to prevent JIT-overwriting
    )
    
    # Add related anchors
    tokyo_desc = "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world."
    platform.long_term.save_knowledge(
        concept="Tokyo", 
        info=tokyo_desc, 
        vector=platform.get_sentence_vector(tokyo_desc), 
        category="Linguistic", 
        label="Wikipedia: City"
    )

    print("   [v9.5 Rescue COMPLETE] Japan Knowledge Shard anchored and locked.")

if __name__ == "__main__":
    fix_japan_knowledge()
