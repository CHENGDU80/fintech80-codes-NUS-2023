import ChatBot from 'react-chatbotify';
import logo from 'assets/images/bot.jpg';

const NUSightGPT = () => {
  const options = {
    tooltip: { mode: 'CLOSE', text: 'Hi!' },
    chatButton: {
      icon: logo
    },
    footer: {
      text: ''
    },
    header: {
      title: <h3 style={{ cursor: 'pointer', margin: 0 }}> SightGPT</h3>,
      showAvatar: true,
      avatar: logo
    },
    notification: {
      disabled: true
    }
  };
  const flow = {
    start: {
      message: () => 'Welcome to SightGPT 🥳! Please upload any files you would like to to analyze.',
      chatDisabled: true,
      file: () => {},
      path: 'question1'
    },
    question1: {
      message: () => `Great! Feel free to ask any questions!`,
      path: 'answer1'
    },
    answer1: {
      message: () => 'The iPad revenue for this quarter is $5.8 billion.',
      path: 'question2'
    },
    question2: {
      message: () => `Ask me another question!`,
      path: 'answer2'
    },
    answer2: {
      message: () => 'The operating cash flow for Apple in this quarter is $26.4 billion.',
      path: 'end'
    }
  };
  return <ChatBot options={options} flow={flow} />;
};

export default NUSightGPT;

// {
//   "q1": {
//       "question": "what is the revenue of ipad this quarter?",
//       "answer": "The iPad revenue for this quarter is $5.8 billion."
//   },
//   "q2": {
//       "question": "what is the operating cash flow for apple in this quarter",
//       "answer": "The operating cash flow for Apple in this quarter is $26.4 billion."
//   }
// }
