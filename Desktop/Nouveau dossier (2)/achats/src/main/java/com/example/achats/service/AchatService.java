package com.example.achats.service;

import com.example.achats.model.Achat;
import com.example.achats.repository.AchatRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class AchatService {
    @Autowired
    private AchatRepository repository;

    public void update(Long id, Achat achat) {
        Optional<Achat> existingAchat = repository.findById(id);
        if (existingAchat.isPresent()) {
            Achat updatedAchat = existingAchat.get();
            updatedAchat.setNumFacture(achat.getNumFacture());
            updatedAchat.setDate(achat.getDate());
            updatedAchat.setFournisseur(achat.getFournisseur());
            updatedAchat.setMatriculeFiscale(achat.getMatriculeFiscale());
            updatedAchat.setLibelle(achat.getLibelle());
            updatedAchat.setMontantHT(achat.getMontantHT());
            updatedAchat.setTva(achat.getTva());
            updatedAchat.setRaS(achat.getRaS());
            updatedAchat.setDroitDeTimbre(achat.getDroitDeTimbre());
            updatedAchat.setRemise(achat.getRemise());
            updatedAchat.setMontantTTC(achat.getMontantTTC());
            updatedAchat.setStatus(achat.getStatus());

            updatedAchat.setProduit(achat.getProduit());
            repository.save(updatedAchat);
        } else {
            throw new RuntimeException("Achat not found with ID: " + id);
        }
    }

    public void ajouter(Achat achat) {
        repository.save(achat);
    }

    public Optional<Achat> rechercheParId(Long id) {
        return repository.findById(id);
    }

    public List<Achat> rechercheTout() {
        return repository.findAll();
    }

    public void supprimer(Long id) {
        repository.deleteById(id);
    }
}
