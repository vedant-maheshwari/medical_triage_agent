// Translation dictionary for English and Hindi
const translations = {
    en: {
        // Header
        appName: "Medical Triage AI",

        // Patient Portal
        pageTitle: "Symptom Assessment",
        pageSubtitle: "Describe your symptoms or upload an image for AI analysis.",

        // Form Labels
        describeSymptoms: "Describe your symptoms",
        symptomPlaceholder: "E.g., Sharp pain in left arm, started 2 hours ago...",
        uploadImage: "Upload Image (Optional)",
        getAssessment: "Get Assessment",
        speakSymptoms: "Speak symptoms",
        listening: "Listening...",

        // Loading
        analyzingSymptoms: "Analyzing symptoms...",

        // Results
        assessmentResult: "Assessment Result",
        recommendedSpecialist: "Recommended Specialist",
        severityLevel: "Severity Level",
        priority: "Priority",
        clinicalNotes: "Clinical Notes",
        recommendedAction: "Recommended Action",
        aiDisclaimer: "💡 This is an AI-generated assessment. A doctor has been notified and will review your case shortly.",
        startNewAssessment: "Start New Assessment",

        // Priority Badges
        critical: "Critical",
        urgent: "Urgent",
        moderate: "Moderate",
        mild: "Mild",

        // Actions
        scheduleAppointment: "Schedule an appointment",
        seekImmediateCare: "Seek immediate medical care",
        visitEmergency: "Visit emergency room immediately",
        consultSpecialist: "Consult with specialist",

        // Errors
        provideInput: "Please provide a description or an image.",
        errorPrefix: "Error: ",

        // Voice Input
        voiceNotSupported: "Voice input not supported in this browser",
        voiceError: "Voice recognition error",
        lowConfidence: "Low recognition quality",

        // Language
        language: "Language",
        english: "English",
        hindi: "हिंदी"
    },

    hi: {
        // Header
        appName: "चिकित्सा ट्राइएज एआई",

        // Patient Portal
        pageTitle: "लक्षण मूल्यांकन",
        pageSubtitle: "अपने लक्षणों का वर्णन करें या एआई विश्लेषण के लिए एक छवि अपलोड करें।",

        // Form Labels
        describeSymptoms: "अपने लक्षणों का वर्णन करें",
        symptomPlaceholder: "उदाहरण: बाएं हाथ में तेज दर्द, 2 घंटे पहले शुरू हुआ...",
        uploadImage: "छवि अपलोड करें (वैकल्पिक)",
        getAssessment: "मूल्यांकन प्राप्त करें",
        speakSymptoms: "लक्षण बोलें",
        listening: "सुन रहे हैं...",

        // Loading
        analyzingSymptoms: "लक्षणों का विश्लेषण कर रहे हैं...",

        // Results
        assessmentResult: "मूल्यांकन परिणाम",
        recommendedSpecialist: "अनुशंसित विशेषज्ञ",
        severityLevel: "गंभीरता स्तर",
        priority: "प्राथमिकता",
        clinicalNotes: "चिकित्सीय टिप्पणियाँ",
        recommendedAction: "अनुशंसित कार्रवाई",
        aiDisclaimer: "💡 यह एक एआई-जनित मूल्यांकन है। एक डॉक्टर को सूचित किया गया है और जल्द ही आपके मामले की समीक्षा करेंगे।",
        startNewAssessment: "नया मूल्यांकन शुरू करें",

        // Priority Badges
        critical: "गंभीर",
        urgent: "तत्काल",
        moderate: "मध्यम",
        mild: "हल्का",

        // Actions
        scheduleAppointment: "अपॉइंटमेंट शेड्यूल करें",
        seekImmediateCare: "तत्काल चिकित्सा देखभाल लें",
        visitEmergency: "तुरंत आपातकालीन कक्ष में जाएं",
        consultSpecialist: "विशेषज्ञ से परामर्श करें",

        // Errors
        provideInput: "कृपया एक विवरण या छवि प्रदान करें।",
        errorPrefix: "त्रुटि: ",

        // Voice Input
        voiceNotSupported: "इस ब्राउज़र में वॉइस इनपुट समर्थित नहीं है",
        voiceError: "वॉइस पहचान त्रुटि",
        lowConfidence: "कम गुणवत्ता की पहचान",

        // Language
        language: "भाषा",
        english: "English",
        hindi: "हिंदी"
    }
};

// Current language (default: English)
let currentLanguage = localStorage.getItem('preferredLanguage') || 'en';

// Get translation for a key
function t(key) {
    return translations[currentLanguage][key] || translations['en'][key] || key;
}

// Set language and update UI
function setLanguage(lang) {
    if (!translations[lang]) {
        console.error(`Language ${lang} not supported`);
        return;
    }

    currentLanguage = lang;
    localStorage.setItem('preferredLanguage', lang);

    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');

        // Update based on element type
        if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
            if (element.hasAttribute('placeholder')) {
                element.placeholder = t(key);
            }
        } else if (element.tagName === 'BUTTON' || element.tagName === 'A') {
            element.textContent = t(key);
        } else {
            element.textContent = t(key);
        }
    });

    // Update language toggle button
    const langToggle = document.getElementById('langToggle');
    if (langToggle) {
        langToggle.textContent = lang === 'en' ? '🌐 EN' : '🌐 हिं';
        langToggle.setAttribute('title', t('language'));
    }

    // Update page title
    document.title = `${t('pageTitle')} - ${t('appName')}`;

    // Update voice recognition language
    if (window.recognition) {
        window.recognition.lang = lang === 'hi' ? 'hi-IN' : 'en-IN';
    }

    // Dispatch event for other components
    document.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang } }));
}

// Initialize language on page load
function initializeLanguage() {
    setLanguage(currentLanguage);
}

// Toggle between languages
function toggleLanguage() {
    const newLang = currentLanguage === 'en' ? 'hi' : 'en';
    setLanguage(newLang);
}
