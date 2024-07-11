import React, { useState } from 'react';
import axios from 'axios';
import styled from '@emotion/styled';

const AppContainer = styled.div`
    text-align: center;
`;

const Header = styled.header`
    background-color: #768bb4;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: calc(10px + 2vmin);
    color: white;
`;

const Form = styled.form`
    margin: 20px 0;
`;

const FileInput = styled.input`
    margin-bottom: 10px;
`;

const SubmitButton = styled.button`
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    background-color: #61dafb;
    border: none;
    border-radius: 5px;
    color: white;
    &:hover {
        background-color: #21a1f1;
    }
`;

const LoadingText = styled.p`
    color: #61dafb;
`;

const ResultText = styled.div`
    margin-top: 20px;
    font-size: 18px;
`;

function App() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setFile(event.target.files[0]);
        }
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!file) {
            alert('Please select a file first!');
            return;
        }

        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://backend:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data.predicted_class);
        } catch (error) {
            console.error('Error uploading file:', error);
            setResult('Error in prediction');
        } finally {
            setLoading(false);
        }
    };

    return (
        <AppContainer>
            <Header>
                <h1>Foraminifera Image Recognition</h1>
                <Form onSubmit={handleSubmit}>
                    <FileInput type="file" onChange={handleFileChange} />
                    <SubmitButton type="submit">Submit</SubmitButton>
                </Form>
                {loading && <LoadingText>Loading...</LoadingText>}
                {result && <ResultText>Result: {result}</ResultText>}
            </Header>
        </AppContainer>
    );
}

export default App;
