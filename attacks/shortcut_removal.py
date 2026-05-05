import re
from attacks.base import Attack
from utils.llm_client import chat

class ShortcutRemovalAttack(Attack):
    """Remove explicit logical/causal connectors using a regex dictionary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shortcuts = [
            "porque", "entonces", "por tanto", "sin embargo", 
            "dado que", "en consecuencia", "luego", "así que", 
            "por eso", "no sólo", "sino", "y así", "en un principio", 
            "de otro lado", "por ello", "debido a", "a causa de"
        ]
        
        # Ordenamos por longitud descendente para que busque frases largas primero 
        sorted_shortcuts = sorted(self.shortcuts, key=len, reverse=True)
        escaped_shortcuts = [re.escape(s) for s in sorted_shortcuts]
        
        self.pattern = re.compile(r'\b(?:' + '|'.join(escaped_shortcuts) + r')\b', re.IGNORECASE)

    def _perturb_text(self, text: str) -> str:
        perturbed = self.pattern.sub("", text)
        
        perturbed = re.sub(r'\.\s*,', '.', perturbed)
        perturbed = re.sub(r'^[,\s]+', '', perturbed)
        
        perturbed = re.sub(r'\s+', ' ', perturbed)
        perturbed = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', perturbed)
        
        perturbed = perturbed.strip()
        if perturbed:
            perturbed = perturbed[0].upper() + perturbed[1:]
            
        return perturbed
