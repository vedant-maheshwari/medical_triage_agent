from app.services import get_user_by_username
from app.auth import verify_password, get_password_hash

print("--- DEBUG LOGIN START ---")
try:
    user = get_user_by_username("doctor")
    if not user:
        print("❌ User 'doctor' not found!")
    else:
        print(f"✅ User found: {user.username}")
        print(f"Stored Hash: {user.hashed_password}")
        
        password = "doctor123"
        is_valid = verify_password(password, user.hashed_password)
        print(f"Testing password '{password}': {is_valid}")
        
        if not is_valid:
            print("❌ Password verification failed!")
            # Try generating a new hash and comparing
            new_hash = get_password_hash(password)
            print(f"New Hash for '{password}': {new_hash}")
            print(f"Verify new hash: {verify_password(password, new_hash)}")
        else:
            print("✅ Password verification SUCCESS!")
except Exception as e:
    print(f"❌ Error: {e}")
print("--- DEBUG LOGIN END ---")
