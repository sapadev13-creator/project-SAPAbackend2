# 🔴 ERROR 403 FORBIDDEN - Twitter API v2 Troubleshooting Guide

## Masalah Anda
```json
{
  "detail": "Server error: 403 Forbidden\nWhen authenticating requests to the Twitter API v2 endpoints, 
  you must use keys and tokens from a Twitter developer App that is 
  attached to a Project. You can create a project via the developer portal."
}
```

---

## 🎯 Root Causes (Pilih yang sesuai)

### ✋ SEBAB #1: Bearer Token URL-Encoded (SUDAH DIPERBAIKI ✅)
**Gejala:**
- Token di .env mengandung `%3D` atau `%2F`
- Example: `...%3DcmUWPZQSLdwGH...` (salah)

**Solusi:**
```env
# ❌ WRONG (URL-encoded)
TWITTER_BEARER_TOKEN=...%3DcmUWPZQSLdwGH...

# ✅ CORRECT (decoded)
TWITTER_BEARER_TOKEN=...=cmUWPZQSLdwGH...
```

**Status:** ✅ SUDAH DIPERBAIKI DI `.env`

---

### ✋ SEBAB #2: Bearer Token Expired
**Gejala:**
- Token valid format, tapi 403 tetap muncul
- Token sudah lama tidak di-regenerate

**Solusi:**

#### Step 1: Buka Twitter Developer Console
```
1. Buka https://developer.twitter.com/en/portal/dashboard
2. Login dengan akun yang sama untuk develop API
```

#### Step 2: Pilih Project & App
```
Di sidebar kiri:
1. Click "Projects & Apps"
2. Select project Anda (contoh: "SAPA OCEAN")
3. Select app Anda (contoh: "SAPA-API-App")
```

#### Step 3: Regenerate Bearer Token
```
Di tab "Keys and tokens":

┌─────────────────────────────────┐
│ Bearer Token                     │
│                                 │
│ [AAAAAAAAAAAAAAAAAAAAAInq6wEA...] │
│                         🔄 Regenerate
└─────────────────────────────────┘

1. Klik tombol "🔄 Regenerate"
2. Confirm regenerate
3. COPY token yang baru (JANGAN dengan %3D atau URL encoding)
```

#### Step 4: Update .env
```bash
TWITTER_BEARER_TOKEN=[PASTE_TOKEN_BARU_DI_SINI]
```

#### Step 5: Restart Server
```bash
# Stop server (Ctrl+C)
# Jalankan lagi
python -m uvicorn sapa_api.app_factory:app --reload
```

---

### ✋ SEBAB #3: App Tidak Attached ke Project (SERING TERJADI!)
**Gejala:**
- Melihat "standalone" app di console
- App tidak di-buat di dalam Project
- Warning: "Legacy standalone app"

**Solusi - Buat App Baru di Project:**

#### Step 1: Go to Developer Console
```
https://developer.twitter.com/en/portal/dashboard
```

#### Step 2: Navigate to Project
```
1. Sidebar → "Projects & Apps"
2. Pilih project yang ada (atau create baru)
```

#### Step 3: Create New App in Project
```
Di halaman project:

┌──────────────────────────────────┐
│ "Apps"                            │
│                                  │
│  [+ Create new app]              │
└──────────────────────────────────┘

1. Klik "+ Create new app"
2. Beri nama: "SAPA-API-App-v2"
3. Klik "Create"
4. Di modal popup, COPY ketiga token:
   - API Key
   - API Secret Key  
   - Bearer Token
```

#### Step 4: Update .env dengan Token Baru
```env
TWITTER_API_KEY=xxxxxxxxxxxxxxxxxxxxx
TWITTER_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWITTER_BEARER_TOKEN=AAAAAAAAAAA...
TWITTER_CLIENT_ID=dXZZYkpfbEZ1eFJE...
TWITTER_CLIENT_SECRET=oPqw4yXuchT2plRO...
```

---

### ✋ SEBAB #4: App Permissions Tidak Sesuai
**Gejala:**
- Token valid, tapi error 403 "Insufficient permissions"
- Baru update permissions tapi belum di-regenerate token

**Solusi:**

#### Step 1: Check Permissions
```
Di Twitter Developer Console → App → "App permissions":

☑️ Read (REQUIRED untuk fetch tweets)
☑️ Write (optional)
☑️ Direct Messages Read (optional)

Pastikan setidaknya "Read" di-check
```

#### Step 2: Regenerate Token Setelah Update Permissions
```
Karena permissions berubah, HARUS regenerate token:

1. Go to "Keys and tokens"
2. Klik 🔄 "Regenerate" di Bearer Token
3. Copy token baru
4. Update .env
5. Restart server
```

---

### ✋ SEBAB #5: OAuth 2.0 Tidak Dikonfigurasi
**Gejala:**
- Error saat login Twitter
- Callback URL tidak valid

**Solusi:**

#### Step 1: Enable OAuth 2.0
```
Di Twitter Developer Console → App → "Authentication settings":

1. Klik "Edit"
2. ☑️ OAuth 2.0 enabled
```

#### Step 2: Set Redirect URI
```
Callback URLs (sesuaikan dengan environment):

Local:
  http://localhost:8000/auth/twitter/callback

Production:
  https://your-domain.com/auth/twitter/callback
```

#### Step 3: Set URLs
```
Website URL:       http://localhost:8000
Terms of service:  http://localhost:8000/terms
Privacy policy:    http://localhost:8000/privacy
```

#### Step 4: Save & Update .env
```env
TWITTER_REDIRECT_URI=http://localhost:8000/auth/twitter/callback
TWITTER_CLIENT_ID=dXZZYkpfbEZ1eFJE...
TWITTER_CLIENT_SECRET=oPqw4yXuchT2plRO...
```

---

## ✅ VERIFICATION CHECKLIST

Sebelum test, pastikan:

- [ ] **Token Format**
  - [ ] Tidak ada `%3D`, `%2F`, atau URL encoding lainnya
  - [ ] Dimulai dengan `AAAAAA...`
  - [ ] Length > 100 characters

- [ ] **App Configuration**
  - [ ] App dibuat DI DALAM Project (bukan standalone)
  - [ ] App permissions includes "Read"
  - [ ] Bearer Token sudah di-regenerate setelah permission change

- [ ] **OAuth 2.0**
  - [ ] OAuth 2.0 enabled
  - [ ] Callback URL set ke: `http://localhost:8000/auth/twitter/callback`
  - [ ] Client ID dan Client Secret di-copy dengan benar

- [ ] **.env File**
  - [ ] Semua 5 credentials ada
  - [ ] Tidak ada URL encoding
  - [ ] File di-save
  - [ ] Server sudah di-restart

---

## 🧪 TEST CREDENTIALS

### Test 1: Validate Token Format
```bash
python twitter_token_validator.py "YOUR_BEARER_TOKEN_HERE"
```

Expected output:
```
✅ Starts with 'AAAAAA': PASS
✅ Length > 100: PASS
✅ Contains only valid chars: PASS
✅ No spaces: PASS
✅ No URL encoding: PASS
✅ TOKEN IS VALID AND READY TO USE!
```

### Test 2: Test Endpoint
```bash
# Restart server dulu
python -m uvicorn sapa_api.app_factory:app --reload
```

Di terminal baru:
```bash
curl -X POST "http://localhost:8000/predict/twitter/profile" \
  -H "Content-Type: application/json" \
  -d '{"profile_url": "https://twitter.com/elonmusk"}'
```

Expected responses:
- ✅ 200 + OCEAN scores = Success!
- ⚠️ 404 "No tweets found" = OK (tweets blocked)
- ❌ 403 "Forbidden" = Token issue
- ❌ 500 "Server error" = Check logs

---

## 📋 .ENV FILE TEMPLATE

Copy template ini dan update dengan credentials Anda:

```env
# ===== TWITTER API v2 CREDENTIALS =====
# Get these from: https://developer.twitter.com/en/portal/dashboard

# From "Keys and tokens" tab:
TWITTER_API_KEY=xxxxxxxxxxxxxxxxxxxxx
TWITTER_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAInq6wEAAAAAcCNTVdT3cqOTQH8cn0XE8vbOapg=cmUWPZQSLdwGH3JRFqbx3pmpVkojuXdom5syjPnT0LqYvLXsZc

# From OAuth 2.0 Client Credentials:
TWITTER_CLIENT_ID=dXZZYkpfbEZ1eFJENUJfLVdVNWI6MTpjaQ
TWITTER_CLIENT_SECRET=oPqw4yXuchT2plROTBQr3mFrV2m-ZjMXyNJecEv_j2iu-IpEB8

# ===== SESSION =====
SESSION_SECRET_KEY=WCaoDWTNnaHNeBVsj_eV6wLqS_XdUD6rrg7D5meAtdU

# ===== REDIRECT URI (sesuaikan dengan environment) =====
TWITTER_REDIRECT_URI=http://localhost:8000/auth/twitter/callback
```

⚠️ **PENTING:**
- Jangan commit `.env` ke repository
- Pastikan `.env` di `.gitignore`
- Ganti credentials dengan yang real

---

## 🚀 QUICK FIX CHECKLIST

Jika masih error 403, ikuti urutan ini:

1. ✅ **Verify token format** (run `twitter_token_validator.py`)
2. ✅ **Regenerate bearer token** di Twitter Console
3. ✅ **Update .env** dengan token baru (tanpa URL encoding)
4. ✅ **Check app permissions** (Read harus enabled)
5. ✅ **Restart server** (`Ctrl+C` dan jalankan lagi)
6. ✅ **Test endpoint** dengan curl

Jika masih error setelah semua langkah di atas:
- [ ] Verify app dibuat DI DALAM Project (bukan standalone)
- [ ] Check OAuth 2.0 configuration
- [ ] Try regenerate token sekali lagi
- [ ] Wait 5-10 minutes untuk API propagation

---

## 📞 REFERENCE LINKS

- Twitter Developer Console: https://developer.twitter.com/en/portal/dashboard
- Twitter API v2 Docs: https://developer.twitter.com/en/docs/twitter-api
- Authentication Guide: https://developer.twitter.com/en/docs/authentication/oauth-2-0
- Project Setup Guide: https://developer.twitter.com/en/docs/projects/overview

---

**Last Updated:** 2026-06-06  
**Status:** Bearer token sudah di-fix di `.env` ✅  
**Next Step:** Ikuti "Test Credentials" section di atas

