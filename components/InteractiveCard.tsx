import React, { useState, useRef, useEffect } from 'react';
import type { FC, ReactNode, MouseEvent } from 'react';
import { GoogleGenAI, Chat, Modality } from '@google/genai';

// FIX: Add type definitions for browser-specific APIs (SpeechRecognition, webkitSpeechRecognition, webkitAudioContext) to resolve TypeScript errors.
// The SpeechRecognition API is not yet part of the standard DOM library for TypeScript, so we define it here.
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onend: (() => void) | null;
  onresult: ((event: any) => void) | null;
  start: () => void;
  stop: () => void;
}

declare global {
  interface Window {
    SpeechRecognition: { new (): SpeechRecognition };
    webkitSpeechRecognition: { new (): SpeechRecognition };
    webkitAudioContext: typeof AudioContext;
  }
}

// --- Helper Icon Components ---

const BoltIcon: FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const GearIcon: FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
);

const BeakerIcon: FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4.5 3h15"></path>
    <path d="M6 3v16a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V3"></path>
    <path d="M6 14h12"></path>
  </svg>
);

const ChatBubbleIcon: FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
);

const MicrophoneIcon: FC = ({ className }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className || "h-6 w-6"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
        <line x1="12" y1="19" x2="12" y2="23"></line>
        <line x1="8" y1="23" x2="16" y2="23"></line>
    </svg>
);


// --- Audio Helper Functions ---
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}


// --- Type Definitions ---

interface ChatMessage {
    role: 'user' | 'model' | 'system';
    content: string;
}

interface TabInfo {
  id: string;
  title: string;
  icon: ReactNode;
}

// --- Content Components ---

const ImplicationsContent: FC = () => (
  <div>
    <h3 className="font-bold text-lg text-slate-100 mb-2">Optimización del Uso de Energía Biológica</h3>
    <p>A lo largo de la evolución, los organismos han desarrollado estructuras, comportamientos y procesos metabólicos que les permiten realizar funciones vitales (como moverse, reproducirse o defenderse) con un menor consumo de energía.</p>
    
    <h3 className="font-bold text-lg text-slate-100 mt-4 mb-2">Selección Natural y Energía</h3>
    <p>La selección natural favorece a los organismos que logran maximizar su rendimiento energético, es decir, aquellos que obtienen más beneficios (como reproducción o supervivencia) por cada unidad de energía invertida.</p>

    <h3 className="font-bold text-lg text-slate-100 mt-4 mb-2">Ejemplos Evolutivos</h3>
    <ul className="list-disc pl-5 space-y-2">
        <li><strong>Locomoción eficiente:</strong> Las aves han desarrollado huesos huecos y músculos especializados que les permiten volar con un gasto energético relativamente bajo.</li>
        <li><strong>Metabolismo adaptado:</strong> Algunos animales hibernan o entran en estados de letargo para conservar energía en ambientes hostiles.</li>
        <li><strong>Cerebros eficientes:</strong> La evolución del cerebro humano ha favorecido estructuras que maximizan el procesamiento de información con un consumo energético relativamente constante.</li>
    </ul>
  </div>
);

const ThermodynamicsContent: FC = () => (
  <div>
    <p className="mb-4">Desde una visión más científica, como plantea la teoría termodinámica de la evolución biológica, los sistemas vivos son sistemas abiertos que intercambian energía y materia con su entorno. Según esta teoría:</p>
    <ul className="list-disc pl-5 space-y-2">
        <li>Los organismos tienden a evolucionar hacia estados de menor potencial energético, buscando un equilibrio con su ambiente.</li>
        <li>La evolución puede entenderse como un proceso de disipación eficiente de energía, donde los sistemas vivos se convierten en mejores transformadores de energía solar o química en trabajo biológico.</li>
    </ul>
  </div>
);

interface ChatInterfaceProps {
    chatHistory: ChatMessage[];
    isLoading: boolean;
    userInput: string;
    isChatReady: boolean;
    isListening: boolean;
    isSpeaking: boolean;
    chatContainerRef: React.RefObject<HTMLDivElement>;
    formRef: React.RefObject<HTMLFormElement>;
    handleSendMessage: (e: React.FormEvent) => void;
    setUserInput: (value: string) => void;
    handleMicClick: () => void;
}

const ChatInterface: FC<ChatInterfaceProps> = ({
    chatHistory,
    isLoading,
    userInput,
    isChatReady,
    isListening,
    isSpeaking,
    chatContainerRef,
    formRef,
    handleSendMessage,
    setUserInput,
    handleMicClick,
}) => (
    <div className="flex flex-col h-full">
        <div ref={chatContainerRef} className="flex-grow overflow-y-auto pr-2 -mr-2 space-y-4">
            {chatHistory.map((msg, index) => (
                <div key={index} className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                    {msg.role === 'model' && (
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                            <BoltIcon />
                        </div>
                    )}
                     <div className={`max-w-[85%] p-3 rounded-lg ${
                        msg.role === 'user' ? 'bg-slate-700 text-slate-200' : 
                        msg.role === 'model' ? 'bg-slate-900/50 text-slate-300' :
                        'bg-red-900/50 text-red-300'
                     }`}>
                        <p className="text-sm" style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                    </div>
                </div>
            ))}
            {isLoading && (
                 <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                        <BoltIcon />
                    </div>
                    <div className="max-w-[85%] p-3 rounded-lg bg-slate-900/50 text-slate-300">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                        </div>
                    </div>
                </div>
            )}
        </div>
        <form ref={formRef} onSubmit={handleSendMessage} className="mt-4 flex gap-2">
            <input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder={isListening ? "Escuchando..." : "Escribe o habla..."}
                disabled={isLoading || !isChatReady || isSpeaking}
                className="flex-grow bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 text-slate-200 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all disabled:opacity-50"
                aria-label="Escribe tu pregunta para el asistente de IA"
            />
            <button
                type="button"
                onClick={handleMicClick}
                disabled={isLoading || !isChatReady || isSpeaking}
                className={`relative flex-shrink-0 w-12 h-10 flex items-center justify-center font-bold px-3 py-2 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed
                    ${isListening 
                        ? 'bg-red-500 text-white' 
                        : 'bg-emerald-500 text-slate-900 hover:bg-emerald-400'
                    } 
                    focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-800`}
                aria-label={isListening ? "Detener grabación" : "Grabar pregunta"}
            >
                {isListening && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>}
                <MicrophoneIcon className="w-5 h-5" />
            </button>
        </form>
    </div>
);


// --- Main Interactive Card Component ---

const InteractiveCard: FC = () => {
  const [activeTab, setActiveTab] = useState<string>('implications');
  const cardRef = useRef<HTMLDivElement>(null);
  
  const [chat, setChat] = useState<Chat | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const aiRef = useRef<GoogleGenAI | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognition | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    const init = async () => {
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
            aiRef.current = ai;

            const chatSession = ai.chats.create({
              model: 'gemini-2.5-flash',
              config: {
                systemInstruction: 'Eres un asistente experto que se especializa en la intersección de la termodinámica, la biología y la evolución. Tu propósito es responder preguntas sobre la eficiencia energética en la evolución con claridad, profundidad y precisión científica. Mantén tus respuestas concisas y atractivas.',
              },
            });
            setChat(chatSession);

            setChatHistory([{
                role: 'model',
                content: '¡Hola! Soy tu asistente de investigación, ejercita tu pensamiento cientifico. Esta es tu oportunidad de profundizar hacia donde la curiosidad te lleve. ¡Sorpréndete de cuánto puedes aprender y descubrir! ¿Qué te gustaría saber?'
            }]);

            // Initialize SpeechRecognition
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if(SpeechRecognition) {
                const recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'es-ES';

                recognition.onresult = (event) => {
                    let finalTranscript = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            finalTranscript += event.results[i][0].transcript;
                        }
                    }
                    if (finalTranscript) {
                        setUserInput(prev => prev + finalTranscript);
                    }
                };
                
                recognition.onend = () => {
                    setIsListening(false);
                    if (userInput.trim()) {
                      formRef.current?.requestSubmit();
                    }
                };

                speechRecognitionRef.current = recognition;
            }

            // Initialize AudioContext
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });

        } catch(error) {
            console.error("Error initializing services:", error);
            setChatHistory([{
                role: 'system',
                content: 'No se pudo inicializar el asistente de IA. Verifica la configuración.'
            }]);
        }
    };
    init();

    return () => {
        speechRecognitionRef.current?.stop();
    }
  }, []);
  
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);
  
  const playAudioResponse = async (text: string) => {
    if (!aiRef.current || !audioContextRef.current) return;
    setIsSpeaking(true);
    try {
        const response = await aiRef.current.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: [{ parts: [{ text }] }],
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: 'Kore' },
                    },
                },
            },
        });
        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if(base64Audio) {
            const audioBuffer = await decodeAudioData(
                decode(base64Audio),
                audioContextRef.current,
                24000,
                1,
            );
            const source = audioContextRef.current.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContextRef.current.destination);
            source.start();
            source.onended = () => {
                setIsSpeaking(false);
            };
        } else {
            setIsSpeaking(false);
        }
    } catch (error) {
        console.error("Error generating audio:", error);
        setIsSpeaking(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading || !chat || isSpeaking) return;

    const userMessage: ChatMessage = { role: 'user', content: userInput };
    setChatHistory(prev => [...prev, userMessage]);
    const currentInput = userInput;
    setUserInput('');
    setIsLoading(true);

    try {
        const response = await chat.sendMessage({ message: currentInput });
        const modelMessage: ChatMessage = { role: 'model', content: response.text };
        setChatHistory(prev => [...prev, modelMessage]);
        await playAudioResponse(response.text);
    } catch (error) {
        console.error("Error sending message:", error);
        const errorMessage: ChatMessage = { role: 'system', content: 'Lo siento, ocurrió un error al procesar tu solicitud.' };
        setChatHistory(prev => [...prev, errorMessage]);
    } finally {
        setIsLoading(false);
    }
  };

  const handleMicClick = () => {
    if (!speechRecognitionRef.current) return;
    if (isListening) {
        speechRecognitionRef.current.stop();
    } else {
        setUserInput(''); // Clear input before starting new recording
        speechRecognitionRef.current.start();
        setIsListening(true);
    }
  };
  
const tabsData: TabInfo[] = [
  {
    id: 'implications',
    title: '¿Qué implica?',
    icon: <GearIcon />,
  },
  {
    id: 'thermodynamics',
    title: 'Perspectiva Termodinámica',
    icon: <BeakerIcon />,
  },
  {
      id: 'chat',
      title: 'Chatea con un experto',
      icon: <ChatBubbleIcon />,
  }
];

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    cardRef.current.style.setProperty('--mouse-x', `${x}px`);
    cardRef.current.style.setProperty('--mouse-y', `${y}px`);
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className="spotlight-card max-w-2xl w-full bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl shadow-2xl shadow-slate-950/50 p-6 md:p-8 transform transition-all duration-300 flex flex-col"
    >
      <header className="flex items-center gap-4 mb-6">
        <div className="flex-shrink-0 bg-emerald-500/10 text-emerald-400 p-3 rounded-full">
          <BoltIcon />
        </div>
        <div>
          <h2 className="text-sm font-medium text-emerald-400 uppercase tracking-widest">Eficiencia Energética en la Evolución</h2>
          <h1 className="text-2xl font-bold text-slate-50 mt-1">LA RESPUESTA</h1>
          <p className="text-slate-400">El Código Secreto del Universo</p>
        </div>
      </header>
      
      <section className="mb-6 text-slate-300 border-t border-b border-slate-700 py-6">
        <p>
          La <strong>"eficiencia energética en la evolución"</strong> se refiere a cómo los organismos vivos han desarrollado mecanismos para utilizar la energía de manera más eficaz a lo largo del tiempo, optimizando su supervivencia y reproducción con el menor gasto energético posible.
        </p>
      </section>

      <nav className="flex flex-wrap border-b border-slate-700">
        {tabsData.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center justify-center gap-2 font-medium px-4 py-3 transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-800 rounded-t-md -mb-px
              ${activeTab === tab.id
                ? 'border-b-2 border-emerald-400 text-emerald-400'
                : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'
              }`}
            aria-selected={activeTab === tab.id}
          >
            <span className="text-emerald-400">{tab.icon}</span>
            <span>{tab.title}</span>
          </button>
        ))}
      </nav>

      <article className="mt-6 min-h-[300px] flex-grow flex flex-col">
          <div key={activeTab} className="content-pane prose prose-invert max-w-none prose-p:text-slate-300 prose-li:text-slate-300 prose-strong:text-slate-200 flex-grow flex flex-col">
              {activeTab === 'implications' && <ImplicationsContent />}
              {activeTab === 'thermodynamics' && <ThermodynamicsContent />}
              {activeTab === 'chat' && (
                <ChatInterface
                    chatHistory={chatHistory}
                    isLoading={isLoading}
                    userInput={userInput}
                    isChatReady={!!chat}
                    isListening={isListening}
                    isSpeaking={isSpeaking}
                    chatContainerRef={chatContainerRef}
                    formRef={formRef}
                    handleSendMessage={handleSendMessage}
                    setUserInput={setUserInput}
                    handleMicClick={handleMicClick}
                />
              )}
          </div>
      </article>
    </div>
  );
};

export default InteractiveCard;
