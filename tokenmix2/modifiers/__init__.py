def get_modifier(method: str, model_type):
    if method == 'enc13':
        if model_type == 'llama':
            from .modify_llama_enc13 import LlamaENC13, Teacher
            return Teacher, LlamaENC13
    elif method == 'enc19':
        if model_type == 'llama':
            from .modify_llama_enc19 import LlamaENC19, Teacher
            return Teacher, LlamaENC19
    elif method == 'enc20':
        if model_type == 'llama':
            from .modify_llama_enc20 import LlamaENC20, Teacher
            return Teacher, LlamaENC20
    elif method == 'enc21':
        if model_type == 'llama':
            from .modify_llama_enc21 import LlamaENC21, Teacher
            return Teacher, LlamaENC21
    elif method == 'hie':
        if model_type == 'llama':
            from .modify_llama_hie import LlamaHierarchical, Teacher
            return Teacher, LlamaHierarchical
    elif method == 'hiedis':
        if model_type == 'llama':
            from .modify_llama_hiedis import LlamaHierarchicalDisdill, Teacher
            return Teacher, LlamaHierarchicalDisdill
    elif method == 'hie2':
        if model_type == 'llama':
            from .modify_llama_hie2 import LlamaHIE2
            return None, LlamaHIE2
    elif method == 'hie3':
        if model_type == 'llama':
            from .modify_llama_hie3 import LlamaHIE3
            return None, LlamaHIE3
    elif method == 'hie5':
        if model_type == 'llama':
            from .modify_llama_hie5 import LlamaHIE5
            return None, LlamaHIE5
    elif method == 'hie6':
        if model_type == 'llama':
            from .modify_llama_hie6 import LlamaHIE6
            return None, LlamaHIE6
    elif method == 'beacons':
        if model_type == 'llama':
            from .modify_llama_beacons import LlamaBeacons
            return None, LlamaBeacons
    elif method == 'arch1':
        if model_type == 'llama':
            from .modify_llama_arch1 import LlamaARCH1
            return None, LlamaARCH1
    elif method == 'arch2':
        if model_type == 'llama':
            from .modify_llama_arch2 import LlamaARCH2
            return None, LlamaARCH2
    elif method == 'arch3':
        if model_type == 'llama':
            from .modify_llama_arch3 import LlamaARCH3
            return None, LlamaARCH3
    elif method == 'arch4':
        if model_type == 'llama':
            from .modify_llama_arch4 import LlamaARCH4
            return None, LlamaARCH4
    elif method == 'arch5':
        if model_type == 'llama':
            from .modify_llama_arch5 import LlamaARCH5
            return None, LlamaARCH5
    elif method == 'arch6':
        if model_type == 'llama':
            from .modify_llama_arch6 import LlamaARCH6
            return None, LlamaARCH6
    elif method == 'arch7':
        if model_type == 'llama':
            from .modify_llama_arch7 import LlamaARCH7
            return None, LlamaARCH7
    elif method == 'arch8':
        if model_type == 'llama':
            from .modify_llama_arch8 import LlamaARCH8
            return None, LlamaARCH8
    elif method == 'arch9':
        if model_type == 'llama':
            from .modify_llama_arch9 import LlamaARCH9
            return None, LlamaARCH9
    elif method == 'archx':
        if model_type == 'llama':
            from .modify_llama_archx import LlamaARCHX
            return None, LlamaARCHX
    elif method == 'arch11':
        if model_type == 'llama':
            from .modify_llama_arch11 import LlamaARCH11
            return None, LlamaARCH11
    elif method == 'arch12':
        if model_type == 'llama':
            from .modify_llama_arch12 import LlamaARCH12
            return None, LlamaARCH12
    elif method == 'arch13':
        if model_type == 'llama':
            from .modify_llama_arch13 import LlamaARCH13
            return None, LlamaARCH13
    elif method == 'arch14':
        if model_type == 'llama':
            from .modify_llama_arch14 import LlamaARCH14
            return None, LlamaARCH14
    elif method == 'arch15':
        if model_type == 'llama':
            from .modify_llama_arch15 import LlamaARCH15
            return None, LlamaARCH15
    elif method == 'arch16':
        if model_type == 'llama':
            from .modify_llama_arch16 import LlamaARCH16
            return None, LlamaARCH16
    elif method == 'arch17':
        if model_type == 'llama':
            from .modify_llama_arch17 import LlamaARCH17
            return None, LlamaARCH17
    elif method == 'arch18':
        if model_type == 'llama':
            from .modify_llama_arch18 import LlamaARCH18
            return None, LlamaARCH18
    elif method == 'arch19':
        if model_type == 'llama':
            from .modify_llama_arch19 import LlamaARCH19
            return None, LlamaARCH19
    elif method == 'arch20':
        if model_type == 'llama':
            from .modify_llama_arch20 import LlamaARCH20
            return None, LlamaARCH20
    elif method == 'arch21':
        if model_type == 'llama':
            from .modify_llama_arch21 import LlamaARCH21
            return None, LlamaARCH21
    elif method == 'arch22':
        if model_type == 'llama':
            from .modify_llama_arch22 import LlamaARCH22
            return None, LlamaARCH22
    elif method == 'hybird1':
        if model_type == 'llama':
            from .modify_llama_hybird1 import LlamaHybird1
            return None, LlamaHybird1
    elif method == 'hybird2':
        if model_type == 'llama':
            from .modify_llama_hybird2 import LlamaHybird2
            return None, LlamaHybird2
    elif method == 'hybird3':
        if model_type == 'llama':
            from .modify_llama_hybird3 import LlamaHybird3
            return None, LlamaHybird3
    elif method == 'hybird4':
        if model_type == 'llama':
            from .modify_llama_hybird4 import LlamaHybird4
            return None, LlamaHybird4
    elif method == 'hybird5':
        if model_type == 'llama':
            from .modify_llama_hybird5 import LlamaHybird5
            return None, LlamaHybird5
    elif method == 'hybird6':
        if model_type == 'llama':
            from .modify_llama_hybird6 import LlamaHybird6
            return None, LlamaHybird6
    elif method == 'hybird7':
        if model_type == 'llama':
            from .modify_llama_hybird7 import LlamaHybird7
            return None, LlamaHybird7
    elif method == 'hybird8':
        if model_type == 'llama':
            from .modify_llama_hybird8 import LlamaHybird8
            return None, LlamaHybird8
    elif method == 'hybird9':
        if model_type == 'llama':
            from .modify_llama_hybird9 import LlamaHybird9
            return None, LlamaHybird9
    elif method == "tinyllama":
        from .modify_tinyllama import TinyLlama
        return None, TinyLlama
    elif method == "origin":
        from .modify_llama_origin import LlamaOrigin
        return None, LlamaOrigin
    elif method == 'flash':
        from .modify_llama_flash import LlamaFlash
        return None, LlamaFlash
    elif method == 'sdpa':
        from .modify_llama_sdpa import LlamaSDPA
        return None, LlamaSDPA
