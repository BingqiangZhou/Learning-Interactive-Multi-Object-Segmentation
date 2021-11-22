from gui import main_window
        
readme_string = """
    operation: 
        mouse: 
            [left button]：interacte
            [right button]：cancel last interactation
        keyboard:
            [number key, include 1-9]: n-th object mark
            ['p' key]: predict result when not in "auto predict" mode
            ['ctrl' + 'alt' + 's' key]：save result inlcude predict label, embedding map(random projection), visual attention map
            ['c' key]: change mode, 'auto predict' or 'press 'p' to predict'
            ['b' key]: change to before image
            ['a' key]: change to after image
    usage:
        
"""

print(readme_string)
main_window()