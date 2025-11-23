const API_URL = "";


// --- Auth Helpers ---
function setToken(token) {
    localStorage.setItem("token", token);
}

function getToken() {
    return localStorage.getItem("token");
}

function removeToken() {
    localStorage.removeItem("token");
}

function isAuthenticated() {
    return !!getToken();
}

async function logout() {
    removeToken();
    window.location.href = "index.html";
}

// --- API Wrapper ---
async function apiCall(endpoint, method = "GET", body = null, auth = false) {
    const headers = {};

    if (auth) {
        const token = getToken();
        if (!token) {
            window.location.href = "login.html";
            return;
        }
        headers["Authorization"] = `Bearer ${token}`;
    }

    const options = {
        method,
        headers
    };

    if (body) {
        if (body instanceof FormData) {
            // FormData handles its own Content-Type
            options.body = body;
        } else {
            headers["Content-Type"] = "application/json";
            options.body = JSON.stringify(body);
        }
    }

    try {
        const response = await fetch(`${API_URL}${endpoint}`, options);

        if (response.status === 401 && auth) {
            removeToken();
            window.location.href = "login.html";
            return;
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "API request failed");
        }

        return await response.json();
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
}

// --- Specific API Functions ---

async function login(username, password) {
    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    const response = await fetch(`${API_URL}/token`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        throw new Error("Login failed");
    }

    const data = await response.json();
    setToken(data.access_token);
    return data;
}

async function getCurrentUser() {
    return apiCall("/users/me", "GET", null, true);
}

async function submitTriage(description, imageFile) {
    const formData = new FormData();
    if (description) formData.append("description", description);
    if (imageFile) formData.append("image", imageFile);

    return apiCall("/triage/assess", "POST", formData);
}

async function getPendingReviews() {
    return apiCall("/triage/pending-reviews", "GET", null, true);
}

async function getCaseDetail(caseId) {
    return apiCall(`/triage/case/${caseId}`, "GET", null, true);
}

async function attendCase(caseId) {
    return apiCall(`/triage/attend/${caseId}`, "POST", null, true);
}

async function submitValidation(validationData) {
    return apiCall("/triage/validate", "POST", validationData, true);
}

async function getAnalytics() {
    return apiCall("/triage/analytics", "GET", null, true);
}

async function getAdminStats() {
    return apiCall("/admin/stats", "GET", null, true);
}
