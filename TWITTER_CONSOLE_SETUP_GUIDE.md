# ⚙️ Twitter API v2 - Setup Step by Step (Console X Terbaru)

## 🔴 ERROR 403 Forbidden - SOLUSI

**Error Message:**
```
403 Forbidden - When authenticating requests to the Twitter API v2 endpoints, 
you must use keys and tokens from a Twitter developer App that is 
attached to a Project. You can create a project via the developer portal.
```

**Penyebab:**
- ❌ Bearer token tidak valid / expired
- ❌ Bearer token tidak dari App yang attached ke Project
- ❌ App permissions tidak sesuai
- ❌ Bearer token URL-encoded (ada `%3D`)

---

## 📋 STEP-BY-STEP SETUP DI TWITTER DEVELOPER CONSOLE

### STEP 1: Login ke Twitter Developer Console
```
1. Buka https://developer.twitter.com/
2. Klik "Sign in" (gunakan Twitter account yang akan develop API)
3. Verifikasi akun Twitter Anda
```

---

### STEP 2: Create atau Navigate ke Project

**Jika belum ada Project:**
```
1. Klik "Create Project" di dashboard
2. Beri nama project (contoh: "SAPA OCEAN")
3. Pilih use case: "Analyze Tweets" atau "Build an app"
4. Pilih environment: "Production"
5. Klik "Continue"
```

**Jika sudah ada Project:**
```
1. Di sidebar, pilih project Anda
2. Lihat "Applications" section
```

---

### STEP 3: Create App di Project

```
1. Di tab "Keys and tokens", klik "Create app"
2. Beri nama app (contoh: "SAPA-API-App")
3. Klik "Create"
4. Di modal popup, copy ketiga token:
   - API Key
   - API Secret Key
   - Bearer Token
```

**⚠️ PENTING:**
- Ketiga token ini HANYA ditampilkan SEKALI
- Simpan di tempat aman (password manager)
- Jangan pernah commit ke repository

---

### STEP 4: Get Authentication Credentials

Di halaman App details, cari section "Keys and tokens":

```
┌─────────────────────────────────────────────┐
│ AUTHENTICATION TOKENS                        │
├─────────────────────────────────────────────┤
│ API Key:                                     │
│  [xxxxxxxxxxxxxxxxxxxxx]    🔄 Regenerate   │
│                                              │
│ API Secret Key:                              │
│  [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]  │
│                                              │
│ Bearer Token:                                │
│  [AAAAAAAAAAAAAAAAAAAA.....................]  │
│                           🔄 Regenerate      │
└─────────────────────────────────────────────┘
```

**Format Bearer Token harus:**
- Diawali dengan `AAAAAA...` (bukan URL-encoded)
- TIDAK boleh ada `%3D` atau simbol URL encoding lainnya

---

### STEP 5: Set App Permissions

```
1. Di halaman App details, cari "App permissions"
2. Pilih "Edit":
   
   ☑️ Read (for reading tweets)
   ☑️ Write (if needed)
   ☑️ Direct Messages Read (optional)

3. Klik "Save"
```

---

### STEP 6: Enable OAuth 2.0

```
1. Scroll ke "Authentication settings"
2. Klik "Edit"
3. Aktifkan:
   ☑️ OAuth 2.0 enabled
   
4. Di "Callback URI", input:
   http://localhost:8000/auth/twitter/callback
   
5. Website URL (optional):
   http://localhost:8000
   
6. Terms of service:
   http://localhost:8000/terms
   
7. Privacy policy:
   http://localhost:8000/privacy

8. Klik "Save"
```

---

### STEP 7: Get OAuth 2.0 Credentials

```
Di tab "OAuth 2.0 Client ID and Client Secret":

┌────────────────────────────────────┐
│ CLIENT CREDENTIALS                  │
├────────────────────────────────────┤
│ Client ID:                          │
│ [dXZZYkpfbEZ1eFJE..........]       │
│                                    │
│ Client Secret:                      │
│ [oPqw4yXuchT2plROTBQr3mF........]  │
└────────────────────────────────────┘
```

Copy:
- **Client ID** → `TWITTER_CLIENT_ID`
- **Client Secret** → `TWITTER_CLIENT_SECRET`

---

## 🔧 UPDATE .env FILE

**STEP 8: Paste Credentials ke .env**

```env
# ===== TWITTER API v2 =====
# Dari App "Keys and tokens" tab:
TWITTER_API_KEY=xxxxxxxxxxxxxxxxxxxxx
TWITTER_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAInq6wEAAAAAcCNTVdT3cqOTQH8cn0XE8vbOapg=cmUWPZQSLdwGH3JRFqbx3pmpVkojuXdom5syjPnT0LqYvLXsZc

# Dari OAuth 2.0 Client Credentials:
TWITTER_CLIENT_ID=dXZZYkpfbEZ1eFJENUJfLVdVNWI6MTpjaQ
TWITTER_CLIENT_SECRET=oPqw4yXuchT2plROTBQr3mFrV2m-ZjMXyNJecEv_j2iu-IpEB8

# Session secret
SESSION_SECRET_KEY=WCaoDWTNnaHNeBVsj_eV6wLqS_XdUD6rrg7D5meAtdU

# Callback URL (sesuaikan dengan server URL)
TWITTER_REDIRECT_URI=http://localhost:8000/auth/twitter/callback
```

**⚠️ PENTING:**
- `TWITTER_BEARER_TOKEN` harus **TIDAK URL-encoded**
- Gunakan format mentah dari developer portal
- Jangan ada `%3D` atau `%2F`

---

## ✅ VERIFY CREDENTIALS

### Langkah Verification di Twitter Console:

```
1. Buka halaman App Anda di Twitter Developer Console
2. Scroll ke section "Authentication"
3. Lihat status:

   ✅ Active - credentials valid
   ❌ Inactive - something wrong
   
4. Jika merah, coba:
   - Regenerate Bearer Token
   - Check permissions
   - Re-authenticate OAuth 2.0
```

---

## 🧪 TEST CREDENTIALS DENGAN CURL

**Test Bearer Token:**
```bash
curl -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  "https://api.twitter.com/2/tweets/search/recent?query=from:twitter&max_results=10"
```

**Expected Response:**
```json
{
  "data": [
    {
      "id": "...",
      "text": "..."
    }
  ]
}
```

---

## 🐛 COMMON ERRORS & SOLUTIONS

### ❌ Error: "Invalid Bearer Token"
**Solusi:**
- Bearer token sudah expired → Regenerate di console
- Format salah → Copy langsung dari console (tidak ada URL encoding)
- Project tidak attached → Buat App baru di Project

### ❌ Error: "403 Forbidden - Project required"
**Solusi:**
- App tidak attached ke Project
- Buat App BARU di dalam Project (Step 3)
- Jangan gunakan legacy standalone App

### ❌ Error: "Insufficient permissions"
**Solusi:**
- Set App permissions ke "Read" (Step 5)
- Regenerate Bearer Token setelah change permissions
- Tunggu 5-10 menit agar changes propagate

### ❌ Error: "Invalid OAuth 2.0 credentials"
**Solusi:**
- Copy Client ID dan Client Secret dengan benar
- Tidak ada whitespace sebelum/sesudah
- Set Callback URI di OAuth settings (Step 6)

---

## 📝 CHECKLIST FINAL

Sebelum test endpoint, pastikan:

- [ ] Sudah create Project di Twitter Developer Console
- [ ] Sudah create App di dalam Project
- [ ] Permissions set ke "Read"
- [ ] OAuth 2.0 enabled dengan Callback URI yang benar
- [ ] Bearer Token di-copy (tidak URL-encoded)
- [ ] Semua credentials di-paste di .env dengan benar
- [ ] Tidak ada URL encoding (`%3D`, `%2F`) di token
- [ ] Bearer Token mulai dengan `AAAAAA...`
- [ ] .env file tidak di-commit ke repository
- [ ] Restart server setelah update .env

---

## 🚀 TEST ENDPOINT SETELAH UPDATE

```bash
# Test di endpoint dengan profile URL
curl -X POST "http://localhost:8000/predict/twitter/profile" \
  -H "Content-Type: application/json" \
  -d '{"profile_url": "https://twitter.com/elonmusk"}'

# Expected: 200 OK dengan OCEAN scores
# Atau: 404 "No tweets found" (jika API rate limited)
# Atau: 500 "Server error: 403" (jika credentials masih salah)
```

---

## 📞 TROUBLESHOOTING TIPS

1. **Check .env file:**
   ```bash
   cat .env  # Verify credentials format
   ```

2. **Check logs saat test:**
   ```bash
   python -m uvicorn sapa_api.app_factory:app --reload
   # Lihat error message di console
   ```

3. **Verify di Twitter Console:**
   - Buka: https://developer.twitter.com/en/portal/dashboard
   - Check Project > App > Keys and tokens
   - Verify Bearer Token aktif

4. **Regenerate Bearer Token:**
   - Di Twitter Console, klik 🔄 Regenerate di Bearer Token
   - Tunggu 30 detik
   - Copy dan update .env
   - Restart server

---

**Last Updated:** 2026-06-06  
**API Version:** Twitter API v2 (X Platform)  
**Reference:** https://developer.twitter.com/

