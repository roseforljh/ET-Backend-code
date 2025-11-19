import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from eztalk_proxy.main import app
    print("SUCCESS: eztalk_proxy.main imported")
    
    # Check routes
    routes = [route.path for route in app.routes]
    print(f"Routes found: {len(routes)}")
    
    if "/everytalk" in routes:
        print("SUCCESS: /everytalk route found")
    else:
        print("FAILURE: /everytalk route NOT found")
        
    if "/everytalk/api/login" in routes:
        print("SUCCESS: /everytalk/api/login route found")
    else:
        print("FAILURE: /everytalk/api/login route NOT found")
        
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()